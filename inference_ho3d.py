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
import torch

from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils

tqdm = partial(_tqdm, bar_format="{desc}: {n_fmt}it [{elapsed}, {rate_fmt}]")

# ── 설정 ───────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train/MDF12"
IMAGE_DIR  = os.path.join(DATA_DIR, "rgb")
MASK_DIR   = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "./output/hold_MDF12_ho3d"

N_VIEWS           = 4
BLUR_THRESHOLD    = 100
VISIBLE_THRESHOLD = 0.4
TARGET_VAL        = 50
OCCLUDE_VAL       = 150
OUTPUT_SIZE       = 512
MARGIN            = 40
DINO_BATCH_SIZE   = 16   # DINOv2 배치 크기
FILTER_WORKERS    = 8    # 필터링 병렬 스레드 수


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
def _check_frame(fname: str) -> tuple[str, float] | None:
    """한 프레임의 블러/visible ratio 계산. 통과 시 (fname, ratio) 반환."""
    img_bgr = cv2.imread(os.path.join(IMAGE_DIR, fname))
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() <= BLUR_THRESHOLD:
        return None

    mask_arr    = np.array(Image.open(os.path.join(MASK_DIR, os.path.splitext(fname)[0] + ".png")))
    obj_pixels  = (mask_arr == TARGET_VAL).sum()
    hand_pixels = (mask_arr == OCCLUDE_VAL).sum()
    if obj_pixels == 0:
        return None
    visible_ratio = obj_pixels / (obj_pixels + hand_pixels + 1e-8)
    if visible_ratio <= VISIBLE_THRESHOLD:
        return None
    return (fname, float(visible_ratio))


def filter_candidates(all_frames: list[str]) -> list[tuple[str, float]]:
    """ThreadPoolExecutor로 병렬 필터링."""
    results = []
    with ThreadPoolExecutor(max_workers=FILTER_WORKERS) as executor:
        for result in tqdm(executor.map(_check_frame, all_frames),
                           total=len(all_frames), desc="Step 1: Filtering"):
            if result is not None:
                results.append(result)
    results.sort(key=lambda x: x[0])
    return results


def extract_features(pipeline, candidate_names: list[str]) -> np.ndarray:
    """이미지 로딩은 병렬, DINOv2는 배치로 처리."""
    def _load(fname):
        return Image.open(os.path.join(IMAGE_DIR, fname)).convert('RGB')

    # 이미지 로딩 병렬화 (I/O 병목 해소)
    with ThreadPoolExecutor(max_workers=FILTER_WORKERS) as executor:
        all_images = list(tqdm(executor.map(_load, candidate_names),
                               total=len(candidate_names), desc="Step 2a: Loading images"))

    # DINOv2 배치 추론 (GPU 활용)
    all_features = []
    for i in tqdm(range(0, len(all_images), DINO_BATCH_SIZE),
                  desc="Step 2b: DINOv2 features"):
        feats = pipeline.encode_image(all_images[i:i + DINO_BATCH_SIZE])
        all_features.append(feats.mean(dim=1).cpu().numpy())
    return np.concatenate(all_features, axis=0)


def select_frames(features: np.ndarray, visible_ratios: np.ndarray,
                  candidate_names: list[str], k: int) -> list[int]:
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
        if visible_ratios[idx] < VISIBLE_THRESHOLD and not_selected:
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
    parser.add_argument("--seq",          type=str, default=None)
    parser.add_argument("--train_root",   type=str, default="/home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train")
    parser.add_argument("--output_root",  type=str, default="./output")
    parser.add_argument("--n_views",      type=int, nargs="+", default=[5, 6, 7, 8])
    args = parser.parse_args()

    seq_name = args.seq if args.seq is not None else os.path.basename(DATA_DIR)
    if args.seq is not None:
        IMAGE_DIR = os.path.join(args.train_root, args.seq, "rgb")
        MASK_DIR  = os.path.join(args.train_root, args.seq, "masks")

    all_frames = sorted(os.listdir(IMAGE_DIR))

    # Step 1: 필터링 (한 번만)
    candidates      = filter_candidates(all_frames)
    candidate_names = [c[0] for c in candidates]
    visible_ratios  = np.array([c[1] for c in candidates])
    print(f"  {len(candidates)} / {len(all_frames)} frames passed filtering")
    assert len(candidates) >= max(args.n_views), "후보 프레임이 너무 적습니다. 필터 조건을 낮추세요."

    # Step 2: 파이프라인 로드 + DINOv2 피처 추출 (한 번만)
    print("Loading pipeline...")
    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.cuda()

    features = extract_features(pipeline, candidate_names)

    # Step 3: n_views별 루프
    for nv in args.n_views:
        print(f"\n{'='*60}")
        print(f"  N_VIEWS = {nv}")
        print(f"{'='*60}")

        output_dir = os.path.join(args.output_root, f"hold_{seq_name}_ho3d_nv{nv}")
        os.makedirs(output_dir, exist_ok=True)

        selected_idx = select_frames(features, visible_ratios, candidate_names, nv)
        selected     = [candidate_names[i] for i in selected_idx]
        print(f"Selected: {selected}")
        print(f"Visible ratios: {[f'{visible_ratios[i]:.3f}' for i in selected_idx]}")

        images, masks = [], []
        for f in tqdm(selected, desc="Preprocessing frames"):
            img_pil, mask_pil = preprocess_frame(
                os.path.join(IMAGE_DIR, f),
                os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"),
            )
            images.append(img_pil)
            masks.append(mask_pil)

        print("Running inference...")
        outputs = pipeline.run_multi_image(
            images, masks,
            seed=1,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 12, "cfg_strength": 3},
        )

        save_outputs(outputs, output_dir)
        print(f"Done. Output saved to {output_dir}/")
        print(f"Frames used: {selected}")
