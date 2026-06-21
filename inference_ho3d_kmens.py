import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Amodal3R'))
from functools import partial
from tqdm import tqdm as _tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['SPCONV_ALGO'] = 'native'

import numpy as np
import imageio
from PIL import Image
import cv2
import trimesh
from sklearn.cluster import KMeans # 추가됨

tqdm = partial(_tqdm, bar_format="{desc}: {n_fmt}it [{elapsed}, {rate_fmt}]")

from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── 설정 ───────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train/ABF12"
IMAGE_DIR  = os.path.join(DATA_DIR, "rgb")
MASK_DIR   = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "./output/hold_ABF12_ho3d"

N_VIEWS           = 4
BLUR_THRESHOLD    = 100   # Laplacian variance 기준
VISIBLE_THRESHOLD = 0.4   # 최소 품질 보장 (필터 기준)
TARGET_VAL        = 50    # 재구성할 객체
OCCLUDE_VAL       = 150   # 가리는 객체 (손)
OUTPUT_SIZE       = 512
MARGIN            = 40

def make_occlude_region(obj_mask: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    """오브젝트 visible 영역의 convex hull 안에 있는 손 픽셀만 occluded로 반환."""
    H, W    = obj_mask.shape
    obj_pts = np.argwhere(obj_mask)[:, ::-1].astype(np.int32)
    if len(obj_pts) < 3:
        return np.zeros((H, W), bool)
    hull      = cv2.convexHull(obj_pts)
    hull_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 1)
    return hand_mask & hull_mask.astype(bool)

def preprocess_frame(img_path: str, mask_path: str):
    """Amodal3R 마스크 생성 + 오브젝트 중심 crop + 512×512 리사이즈."""
    image_np = np.array(Image.open(img_path).convert("RGB"))
    mask_np  = np.array(Image.open(mask_path))
    H, W     = mask_np.shape[:2]

    obj_mask    = mask_np == TARGET_VAL
    hand_region = mask_np == OCCLUDE_VAL

    amodal_mask = np.full((H, W), 255, dtype=np.uint8)
    amodal_mask[obj_mask] = 200
    amodal_mask[make_occlude_region(obj_mask, hand_region)] = 0

    ys, xs = np.where(obj_mask)
    if len(ys) == 0:
        return None, None
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half   = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + MARGIN
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)

    image_crop = image_np[y0:y1, x0:x1]
    mask_crop  = amodal_mask[y0:y1, x0:x1]

    size     = max(image_crop.shape[:2])
    pad_img  = np.ones((size, size, 3), dtype=np.uint8) * 255
    pad_mask = np.ones((size, size),    dtype=np.uint8) * 255
    h, w     = image_crop.shape[:2]
    oh, ow   = (size - h) // 2, (size - w) // 2
    pad_img [oh:oh+h, ow:ow+w] = image_crop
    pad_mask[oh:oh+h, ow:ow+w] = mask_crop

    image_pil = Image.fromarray(pad_img).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    mask_pil  = Image.fromarray(pad_mask).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST).convert("L")
    return image_pil, mask_pil

def filter_candidates(image_dir: str, mask_dir: str) -> list[tuple[str, float]]:
    """블러 + visible ratio 기준으로 후보 프레임 필터링."""
    all_frames = sorted(os.listdir(image_dir))
    candidates = []
    for fname in tqdm(all_frames, desc="Step 1: Filtering"):
        img_bgr = cv2.imread(os.path.join(image_dir, fname))
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() <= BLUR_THRESHOLD:
            continue

        mask_fname = os.path.splitext(fname)[0] + ".png"
        mask_arr   = np.array(Image.open(os.path.join(mask_dir, mask_fname)))
        obj_pixels  = (mask_arr == TARGET_VAL).sum()
        hand_pixels = (mask_arr == OCCLUDE_VAL).sum()
        if obj_pixels == 0:
            continue
        visible_ratio = obj_pixels / (obj_pixels + hand_pixels + 1e-8)
        if visible_ratio <= VISIBLE_THRESHOLD:
            continue

        candidates.append((fname, float(visible_ratio)))

    print(f"  {len(candidates)} / {len(all_frames)} frames passed filtering")
    assert len(candidates) >= N_VIEWS, "후보 프레임이 너무 적습니다. 필터 조건을 낮추세요."
    return candidates


def extract_features(pipeline, image_dir: str, mask_dir: str, candidate_names: list[str]) -> np.ndarray:
    """[수정됨] 배경을 마스킹한 후 객체 중심의 DINOv2 피처 추출."""
    features = []
    for fname in tqdm(candidate_names, desc="Step 2: Masked DINOv2 features"):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")
        
        img_np = np.array(Image.open(img_path).convert('RGB'))
        mask_np = np.array(Image.open(mask_path))
        
        # 타겟 객체만 추출하고 배경 및 가림(손)은 흰색으로 처리
        obj_mask = (mask_np == TARGET_VAL)
        img_masked = img_np.copy()
        img_masked[~obj_mask] = 255 
        
        img_pil = Image.fromarray(img_masked)
        feat = pipeline.encode_image([img_pil])  # (1, N_patches, D)
        features.append(feat.mean(dim=1).squeeze(0).cpu().numpy())
    return np.stack(features)


def select_frames(features: np.ndarray, visible_ratios: np.ndarray,
                  candidate_names: list[str], k: int) -> list[int]:
    """[수정됨] FPS 대신 K-Means 기반 메도이드(Medoid) 선택 알고리즘 적용."""
    # 1. K-Means 클러스터링으로 뷰를 k개의 군집으로 분할
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(features)

    selected_idx = []
    for i in tqdm(range(k), desc="Step 3: K-Medoids Clustering"):
        # 2. 각 클러스터의 중심점(Center) 가져오기
        center = kmeans.cluster_centers_[i]
        # 3. 중심점과 모든 프레임 피처 간의 거리 계산
        distances = np.linalg.norm(features - center, axis=1)
        
        # 4. 중심에 가장 가까운 프레임부터 확인하며 가시성 기준 충족 시 선택
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices:
            # 중복 방지 및 퀄리티 보장
            if idx not in selected_idx and visible_ratios[idx] >= VISIBLE_THRESHOLD:
                selected_idx.append(idx)
                break
        else:
            # 예외 처리: 모든 조건 미달 시 가장 가까운 미선택 프레임 강제 추가
            for idx in sorted_indices:
                if idx not in selected_idx:
                    selected_idx.append(idx)
                    break

    return sorted(selected_idx)

def save_outputs(outputs: dict, output_dir: str) -> None:
    video_gs   = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0],     bg_color=(1, 1, 1))['normal']
    video = [np.concatenate([fg, fm], axis=1) for fg, fm in zip(video_gs, video_mesh)]
    imageio.mimsave(os.path.join(output_dir, "output.gif"), video, fps=30)

    gaussian = outputs['gaussian'][0]
    mv_gs, _, _ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1))
    for i, frame in enumerate(tqdm(mv_gs['color'], desc="Saving GS renders")):
        cv2.imwrite(os.path.join(output_dir, f"{i:03d}_gs.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    mesh = outputs['mesh'][0]
    mv_mesh, _, _ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
    for i, frame in enumerate(tqdm(mv_mesh['normal'], desc="Saving mesh renders")):
        cv2.imwrite(os.path.join(output_dir, f"{i:03d}_mesh.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
    faces    = mesh.faces.cpu().numpy()    if hasattr(mesh.faces, 'cpu')    else mesh.faces
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
        os.path.join(output_dir, "mesh.ply")
    )


if __name__ == "__main__":
    # Step 1: 품질 필터링
    candidates      = filter_candidates(IMAGE_DIR, MASK_DIR)
    candidate_names = [c[0] for c in candidates]
    visible_ratios  = np.array([c[1] for c in candidates])

    # Step 2: 파이프라인 로드 + DINOv2 피처 추출
    print("Loading pipeline...")
    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.cuda()

    # extract_features 호출 시 MASK_DIR 추가
    features = extract_features(pipeline, IMAGE_DIR, MASK_DIR, candidate_names)

    # Step 3: 클러스터링 기반 프레임 선택
    selected_idx = select_frames(features, visible_ratios, candidate_names, N_VIEWS)
    selected     = [candidate_names[i] for i in selected_idx]
    print(f"Selected: {selected}")
    print(f"Visible ratios: {[f'{visible_ratios[i]:.3f}' for i in selected_idx]}")

    # 전처리
    images, masks = [], []
    for f in tqdm(selected, desc="Preprocessing frames"):
        img_pil, mask_pil = preprocess_frame(
            os.path.join(IMAGE_DIR, f),
            os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"),
        )
        images.append(img_pil)
        masks.append(mask_pil)

    # 추론
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Running inference...")
    outputs = pipeline.run_multi_image(
        images, masks,
        seed=1,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
    )

    # 저장
    save_outputs(outputs, OUTPUT_DIR)
    print(f"\nDone. Output saved to {OUTPUT_DIR}/")
    print(f"Frames used: {selected}")