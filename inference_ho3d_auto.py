import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Amodal3R'))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['SPCONV_ALGO'] = 'native'

import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm as _tqdm

import numpy as np
import imageio
from PIL import Image
import cv2
import trimesh

from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils

tqdm = partial(_tqdm, bar_format="{desc}: {n_fmt}it [{elapsed}, {rate_fmt}]")

# ── 고정 설정 (시퀀스와 무관) ───────────────────────────────────────────────────
DATA_DIR   = "/home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train/ABF12"
IMAGE_DIR  = os.path.join(DATA_DIR, "rgb")
MASK_DIR   = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "./output/hold_ABF12_ho3d"

N_VIEWS         = 4
TARGET_VAL      = 50
OCCLUDE_VAL     = 150
OUTPUT_SIZE     = 512
MARGIN          = 40
DINO_BATCH_SIZE = 16
FILTER_WORKERS  = 8
MIN_CANDIDATES  = N_VIEWS * 10   # 후보 최소 수 (adaptive threshold 기준)


# ── 마스크/이미지 전처리 ────────────────────────────────────────────────────────
def make_occlude_region(obj_mask: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
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


# ── 프레임 선택 ─────────────────────────────────────────────────────────────────
def _compute_frame_stats(fname: str) -> tuple[str, float, float]:
    """(fname, blur_var, visible_ratio) 반환. obj 없으면 vis=0."""
    img_bgr = cv2.imread(os.path.join(IMAGE_DIR, fname))
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur    = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mask_arr    = np.array(Image.open(os.path.join(MASK_DIR, os.path.splitext(fname)[0] + ".png")))
    obj_pixels  = (mask_arr == TARGET_VAL).sum()
    hand_pixels = (mask_arr == OCCLUDE_VAL).sum()
    vis = float(obj_pixels / (obj_pixels + hand_pixels + 1e-8)) if obj_pixels > 0 else 0.0
    return (fname, blur, vis)


def adaptive_thresholds(blur_vals: np.ndarray, vis_vals: np.ndarray) -> tuple[float, float]:
    """시퀀스 분포 기반으로 blur/visible 임계값 자동 결정.

    - blur:    하위 10% 제거 (심하게 흔들린 프레임만 탈락)
    - visible: 후보가 MIN_CANDIDATES 이상 남도록 percentile을 완화
    """
    blur_thresh = float(np.percentile(blur_vals, 10))

    valid_vis = vis_vals[vis_vals > 0]
    vis_thresh = 0.0
    for pct in range(40, -1, -5):
        t = float(np.percentile(valid_vis, pct)) if len(valid_vis) else 0.0
        n = int(((blur_vals >= blur_thresh) & (vis_vals > t)).sum())
        if n >= MIN_CANDIDATES:
            vis_thresh = t
            break

    return blur_thresh, vis_thresh


def filter_candidates(all_frames: list[str]) -> tuple[list[tuple[str, float]], float, float]:
    """전체 통계 분석 → 적응형 임계값 → 필터링."""
    with ThreadPoolExecutor(max_workers=FILTER_WORKERS) as executor:
        stats = list(tqdm(executor.map(_compute_frame_stats, all_frames),
                          total=len(all_frames), desc="Step 1: Analyzing"))

    blur_vals = np.array([s[1] for s in stats])
    vis_vals  = np.array([s[2] for s in stats])

    blur_thresh, vis_thresh = adaptive_thresholds(blur_vals, vis_vals)
    print(f"  Adaptive thresholds — blur≥{blur_thresh:.1f}  visible>{vis_thresh:.3f}")

    candidates = [(s[0], s[2]) for s in stats
                  if s[1] >= blur_thresh and s[2] > vis_thresh]
    candidates.sort(key=lambda x: x[0])
    return candidates, blur_thresh, vis_thresh


def extract_features(pipeline, candidate_names: list[str]) -> np.ndarray:
    """이미지 로딩은 병렬, DINOv2는 배치로 처리."""
    def _load(fname):
        return Image.open(os.path.join(IMAGE_DIR, fname)).convert('RGB')

    with ThreadPoolExecutor(max_workers=FILTER_WORKERS) as executor:
        all_images = list(tqdm(executor.map(_load, candidate_names),
                               total=len(candidate_names), desc="Step 2a: Loading images"))

    all_features = []
    for i in tqdm(range(0, len(all_images), DINO_BATCH_SIZE),
                  desc="Step 2b: DINOv2 features"):
        feats = pipeline.encode_image(all_images[i:i + DINO_BATCH_SIZE])
        all_features.append(feats.mean(dim=1).cpu().numpy())
    return np.concatenate(all_features, axis=0)


def select_frames(features: np.ndarray, visible_ratios: np.ndarray,
                  candidate_names: list[str], k: int,
                  vis_thresh: float) -> list[int]:
    """DINOv2 FPS로 k개 선택 후 visible_ratio 최소 기준 보장."""
    centroid     = features.mean(axis=0)
    anchor       = int(np.sum((features - centroid) ** 2, axis=1).argmax())
    selected_idx = [anchor]
    dists        = np.full(len(features), np.inf)

    for _ in tqdm(range(k - 1), desc="Step 3: FPS"):
        d     = np.sum((features - features[selected_idx[-1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        dists[selected_idx] = -np.inf
        selected_idx.append(int(np.argmax(dists)))

    not_selected = sorted(
        [i for i in range(len(features)) if i not in selected_idx],
        key=lambda i: -visible_ratios[i]
    )
    for pos, idx in enumerate(selected_idx):
        if visible_ratios[idx] <= vis_thresh and not_selected:
            replacement = not_selected.pop(0)
            print(f"  [교체] {candidate_names[idx]} (vis={visible_ratios[idx]:.3f})"
                  f" → {candidate_names[replacement]} (vis={visible_ratios[replacement]:.3f})")
            selected_idx[pos] = replacement

    return sorted(selected_idx)


# ── 결과 저장 ───────────────────────────────────────────────────────────────────
def save_outputs(outputs: dict, output_dir: str) -> None:
    video_gs   = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0],     bg_color=(1, 1, 1))['normal']
    video = [np.concatenate([fg, fm], axis=1) for fg, fm in zip(video_gs, video_mesh)]
    imageio.mimsave(os.path.join(output_dir, "output.gif"), video, fps=30)

    gaussian    = outputs['gaussian'][0]
    mv_gs, _, _ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1))
    for i, frame in enumerate(tqdm(mv_gs['color'], desc="Saving GS renders")):
        cv2.imwrite(os.path.join(output_dir, f"{i:03d}_gs.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    mesh          = outputs['mesh'][0]
    mv_mesh, _, _ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
    for i, frame in enumerate(tqdm(mv_mesh['normal'], desc="Saving mesh renders")):
        cv2.imwrite(os.path.join(output_dir, f"{i:03d}_mesh.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices
    faces    = mesh.faces.cpu().numpy()    if hasattr(mesh.faces, 'cpu')    else mesh.faces
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
        os.path.join(output_dir, "mesh.ply")
    )


# ── 메인 ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",         type=str, default=None)
    parser.add_argument("--train_root",  type=str, default="/home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train")
    parser.add_argument("--output_root", type=str, default="./output")
    args = parser.parse_args()

    if args.seq is not None:
        IMAGE_DIR  = os.path.join(args.train_root, args.seq, "rgb")
        MASK_DIR   = os.path.join(args.train_root, args.seq, "masks")
        OUTPUT_DIR = os.path.join(args.output_root, f"hold_{args.seq}_ho3d_auto")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_frames = sorted(os.listdir(IMAGE_DIR))

    # Step 1: 통계 분석 + 적응형 필터링
    candidates, blur_thresh, vis_thresh = filter_candidates(all_frames)
    candidate_names = [c[0] for c in candidates]
    visible_ratios  = np.array([c[1] for c in candidates])
    print(f"  {len(candidates)} / {len(all_frames)} frames passed filtering")
    assert len(candidates) >= N_VIEWS, "후보 프레임이 너무 적습니다."

    # Step 2: 파이프라인 로드 + 배치 DINOv2 피처 추출
    print("Loading pipeline...")
    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.cuda()

    features = extract_features(pipeline, candidate_names)

    # Step 3: FPS 프레임 선택
    selected_idx = select_frames(features, visible_ratios, candidate_names, N_VIEWS, vis_thresh)
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
