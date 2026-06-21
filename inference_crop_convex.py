import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Amodal3R'))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import json
import pickle
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm as _tqdm
from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils
import cv2
import trimesh

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

tqdm = partial(_tqdm, bar_format="{desc}: {n_fmt}it [{elapsed}, {rate_fmt}]")


def extract_glb(gs, mesh, mesh_simplify=0.95, texture_size=1024, export_path="output.glb"):
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb.export(export_path)
    return export_path


def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs
    mesh.export(filename)


def make_occlude_region(obj_mask: np.ndarray, hand_mask: np.ndarray,
                        method: str = "convex", dilate_px: int = 10) -> np.ndarray:
    if method == "dilate":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        region = cv2.dilate(obj_mask.astype(np.uint8), kernel).astype(bool)
        return hand_mask & region
    elif method == "flood":
        # 오브젝트에 인접한 손 픽셀을 seed로, 연결된 손 컴포넌트 전체를 occluded 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obj_dilated = cv2.dilate(obj_mask.astype(np.uint8), kernel).astype(bool)
        seed_hand = hand_mask & obj_dilated
        _, labels = cv2.connectedComponents(hand_mask.astype(np.uint8))
        seed_labels = set(labels[seed_hand].tolist()) - {0}
        result = np.zeros(hand_mask.shape, bool)
        for lbl in seed_labels:
            result |= (labels == lbl)
        return result & hand_mask
    else:  # convex
        H, W = obj_mask.shape
        obj_pts = np.argwhere(obj_mask)[:, ::-1].astype(np.int32)
        if len(obj_pts) < 3:
            return np.zeros((H, W), bool)
        hull = cv2.convexHull(obj_pts)
        hull_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull], 1)
        return hand_mask & hull_mask.astype(bool)


def preprocess_frame(rgb_path, mask_path, target_val=50, occlude_val=150, margin=40, output_size=512,
                     black_bg=False, occlude_method="convex", dilate_px=10, no_crop=False):
    image_np = np.array(Image.open(rgb_path).convert("RGB"))
    mask_np = np.array(Image.open(mask_path))
    H, W = mask_np.shape[:2]

    obj_mask = mask_np == target_val
    hand_mask = mask_np == occlude_val

    amodal_mask = np.full((H, W), 255, dtype=np.uint8)
    amodal_mask[obj_mask] = 200
    amodal_mask[make_occlude_region(obj_mask, hand_mask, method=occlude_method, dilate_px=dilate_px)] = 0

    if black_bg:
        image_np[amodal_mask == 255] = 0

    ys, xs = np.where(obj_mask)
    if len(ys) == 0:
        return None, None

    if no_crop:
        image_region = image_np
        mask_region = amodal_mask
    else:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = max(y1 - y0, x1 - x0) // 2 + margin
        y0c = max(0, cy - half); y1c = min(H, cy + half)
        x0c = max(0, cx - half); x1c = min(W, cx + half)
        image_region = image_np[y0c:y1c, x0c:x1c]
        mask_region = amodal_mask[y0c:y1c, x0c:x1c]

    size = max(image_region.shape[:2])
    img_bg = 0 if black_bg else 255
    pad_img = np.full((size, size, 3), img_bg, dtype=np.uint8)
    pad_mask = np.ones((size, size), dtype=np.uint8) * 255
    h, w = image_region.shape[:2]
    oh, ow = (size - h) // 2, (size - w) // 2
    pad_img[oh:oh+h, ow:ow+w] = image_region
    pad_mask[oh:oh+h, ow:ow+w] = mask_region

    image_pil = Image.fromarray(pad_img).resize((output_size, output_size), Image.LANCZOS)
    mask_pil = Image.fromarray(pad_mask).resize((output_size, output_size), Image.NEAREST).convert("L")
    return image_pil, mask_pil


# ── 후보 필터링 ──────────────────────────────────────────────────────────────────

def _check_frame(fname, image_dir, mask_dir, target_val, occlude_val, blur_threshold, visible_threshold):
    img_bgr = cv2.imread(os.path.join(image_dir, fname))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() <= blur_threshold:
        return None

    mask_arr = np.array(Image.open(os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")))
    obj_pixels = (mask_arr == target_val).sum()
    hand_pixels = (mask_arr == occlude_val).sum()
    if obj_pixels == 0:
        return None

    obj_mask = mask_arr == target_val
    H, W = obj_mask.shape
    if obj_mask[0, :].any() or obj_mask[H-1, :].any() or obj_mask[:, 0].any() or obj_mask[:, W-1].any():
        return None

    visible_ratio = obj_pixels / (obj_pixels + hand_pixels + 1e-8)
    if visible_ratio <= visible_threshold:
        return None
    return (fname, float(visible_ratio))


def filter_candidates(all_frames, image_dir, mask_dir, target_val, occlude_val,
                      blur_threshold, visible_threshold, n_workers=8):
    check = partial(_check_frame, image_dir=image_dir, mask_dir=mask_dir,
                    target_val=target_val, occlude_val=occlude_val,
                    blur_threshold=blur_threshold, visible_threshold=visible_threshold)
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(check, all_frames),
                           total=len(all_frames), desc="Step 1: Filtering"):
            if result is not None:
                results.append(result)
    results.sort(key=lambda x: x[0])
    return results


# ── FPS 선택 ─────────────────────────────────────────────────────────────────────

def extract_features(pipeline, candidate_names, image_dir, n_workers=8, batch_size=16):
    def _load(fname):
        return Image.open(os.path.join(image_dir, fname)).convert('RGB')

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        all_images = list(tqdm(executor.map(_load, candidate_names),
                               total=len(candidate_names), desc="Step 2a: Loading images"))

    all_features = []
    for i in tqdm(range(0, len(all_images), batch_size), desc="Step 2b: DINOv2 features"):
        feats = pipeline.encode_image(all_images[i:i + batch_size])
        all_features.append(feats.mean(dim=1).cpu().numpy())
    return np.concatenate(all_features, axis=0)


def remove_outlier_views(features: np.ndarray, candidate_names: list[str],
                         visible_ratios: np.ndarray, k: int,
                         sigma: float = 1.5) -> tuple[np.ndarray, list, np.ndarray]:
    """Cosine similarity 기반으로 다수 뷰와 너무 다른 아웃라이어 프레임을 제거.

    mean - sigma*std 이하인 프레임을 아웃라이어로 제거. sigma=0 이면 비활성화.
    """
    if sigma <= 0 or len(candidate_names) <= k:
        return features, candidate_names, visible_ratios

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    feats_norm = features / norms
    sim_matrix = feats_norm @ feats_norm.T  # (N, N) cosine similarity
    np.fill_diagonal(sim_matrix, np.nan)
    mean_sim = np.nanmean(sim_matrix, axis=1)

    threshold = mean_sim.mean() - sigma * mean_sim.std()
    keep = mean_sim >= threshold

    if keep.sum() < k:
        keep = np.zeros(len(features), bool)
        keep[np.argsort(mean_sim)[-k:]] = True

    removed = [(candidate_names[i], float(mean_sim[i])) for i in range(len(candidate_names)) if not keep[i]]
    if removed:
        for name, sim in removed:
            print(f"  [아웃라이어 제거] {name} (mean_sim={sim:.3f}, threshold={threshold:.3f})")
    else:
        print(f"  [아웃라이어 없음] mean_sim 범위: {mean_sim.min():.3f}~{mean_sim.max():.3f}, threshold={threshold:.3f}")

    keep_idx = np.where(keep)[0]
    return features[keep_idx], [candidate_names[i] for i in keep_idx], visible_ratios[keep_idx]


def select_frames_fps(features, visible_ratios, candidate_names, k, visible_threshold):
    centroid = features.mean(axis=0)
    anchor = int(np.sum((features - centroid) ** 2, axis=1).argmax())
    selected_idx = [anchor]
    dists = np.full(len(features), np.inf)

    for _ in tqdm(range(k - 1), desc="Step 3: FPS"):
        d = np.sum((features - features[selected_idx[-1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        dists[selected_idx] = -np.inf
        selected_idx.append(int(np.argmax(dists)))

    not_selected = sorted( 
        [i for i in range(len(features)) if i not in selected_idx],
        key=lambda i: -visible_ratios[i]
    )
    for pos, idx in enumerate(selected_idx):
        if visible_ratios[idx] < visible_threshold and not_selected:
            replacement = not_selected.pop(0)
            print(f"  [교체] {candidate_names[idx]} (vis={visible_ratios[idx]:.3f})"
                  f" → {candidate_names[replacement]} (vis={visible_ratios[replacement]:.3f})")
            selected_idx[pos] = replacement

    return sorted(selected_idx)


# ── Rotation 다양성 기반 선택 ─────────────────────────────────────────────────────

def select_frames_rotation(candidate_names, seq_dir, k):
    """meta pkl의 objRot을 9D flatten해서 FPS로 자세 다양성 최대화."""
    meta_dir = os.path.join(seq_dir, "meta")

    valid_names = []
    features = []
    for fname in tqdm(candidate_names, desc="Step 2: Loading rotations"):
        stem = os.path.splitext(fname)[0]
        with open(os.path.join(meta_dir, stem + ".pkl"), 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        obj_rot = meta.get('objRot')
        if obj_rot is None:
            continue
        R = cv2.Rodrigues(obj_rot)[0] if obj_rot.shape != (3, 3) else obj_rot
        valid_names.append(fname)
        features.append(R.flatten())  # 9D rotation feature
    features = np.array(features)  # (N, 9)

    # FPS: L2 거리 기준으로 가장 다양한 자세 k개 선택
    centroid = features.mean(axis=0)
    anchor = int(np.sum((features - centroid) ** 2, axis=1).argmax())
    selected_idx = [anchor]
    dists = np.full(len(features), np.inf)

    for _ in tqdm(range(k - 1), desc="Step 3: Rotation FPS"):
        d = np.sum((features - features[selected_idx[-1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        dists[selected_idx] = -np.inf
        selected_idx.append(int(np.argmax(dists)))

    return sorted([valid_names[i] for i in selected_idx])


# ── 그리드 저장 ──────────────────────────────────────────────────────────────────

def save_grid(pil_list, save_path, ncols=4):
    n = len(pil_list)
    w, h = pil_list[0].size
    nrows = (n + ncols - 1) // ncols
    grid = Image.new("RGB", (ncols * w, nrows * h), (255, 255, 255))
    for i, img in enumerate(pil_list):
        grid.paste(img.convert("RGB"), ((i % ncols) * w, (i // ncols) * h))
    grid.save(save_path)


# ── 배치 모드 헬퍼 ────────────────────────────────────────────────────────────────

def _save_outputs(pipeline, images, masks, out_dir, seq_dir, used_ids,
                  select_method, n_views, occlude_method, black_bg, outlier_sigma, args):
    os.makedirs(out_dir, exist_ok=True)
    save_grid(images, os.path.join(out_dir, "input_images.png"))
    save_grid(masks,  os.path.join(out_dir, "input_masks.png"))

    outputs = pipeline.run_multi_image(
        images, masks, seed=1,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
    )

    video_gs   = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0],     bg_color=(1, 1, 1))['normal']
    video = [np.concatenate([a, b], axis=1) for a, b in zip(video_gs, video_mesh)]
    imageio.mimsave(os.path.join(out_dir, "sample_multi.gif"), video, fps=30)

    gaussian = outputs['gaussian'][0]
    mv_gs, _, _ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1))
    for i, frame in enumerate(mv_gs['color']):
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_gs.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    mesh = outputs['mesh'][0]
    mv_mesh, _, _ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
    for i, frame in enumerate(mv_mesh['normal']):
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_mesh.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    save_mesh(mesh, os.path.join(out_dir, "mesh.ply"))

    config = {
        "seq_dir": seq_dir,
        "frames": used_ids,
        "mode": "auto_select",
        "select_method": select_method,
        "n_views": n_views,
        "occlude_method": occlude_method,
        "dilate_px": args.dilate_px,
        "no_crop": args.no_crop,
        "black_bg": black_bg,
        "outlier_sigma": outlier_sigma,
        "blur_threshold": args.blur_threshold,
        "visible_threshold": args.visible_threshold,
        "margin": args.margin,
        "output_size": args.output_size,
    }
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump(config, f, indent=2)


def _run_batch(args, pipeline):
    """배치 모드: 파이프라인 1회 로드 + 시퀀스당 DINOv2 피처 1회 추출.

    제거된 중복 연산 (vs 5832회 개별 실행):
      파이프라인 로드  : 5832회 → 1회
      후보 필터링      : 972회  → 54회 (시퀀스당 1회)
      DINOv2 피처 추출 : 648회  → 54회 (시퀀스당 1회)
      프레임 선택      : (n_views, outlier_sigma)당 1회 → occlude/bg 조합에 재사용
    """
    ALL_SELECT_METHODS  = ["fps"]
    ALL_OCCLUDE_METHODS = ["convex"]
    ALL_BLACK_BG        = [False, True]
    ALL_OUTLIER_SIGMAS  = [None, 1.5]   # fps 전용

    n_views_list = args.n_views if args.n_views else [4, 6, 8]

    seq_dirs = sorted(
        os.path.join(args.train_dir, d)
        for d in os.listdir(args.train_dir)
        if os.path.isdir(os.path.join(args.train_dir, d))
    )

    fps_per_nv   = len(ALL_OUTLIER_SIGMAS) * len(ALL_OCCLUDE_METHODS) * len(ALL_BLACK_BG)
    rot_per_nv   = len(ALL_OCCLUDE_METHODS) * len(ALL_BLACK_BG)
    runs_per_seq = len(n_views_list) * (fps_per_nv + rot_per_nv)
    total        = len(seq_dirs) * runs_per_seq
    print(f"Sequences : {len(seq_dirs)}")
    print(f"n_views   : {n_views_list}")
    print(f"Total runs: {total}")
    print("=" * 60)

    run = skipped = failed = 0

    for seq_dir in seq_dirs:
        seq_name  = os.path.basename(seq_dir)
        image_dir = os.path.join(seq_dir, "rgb")
        mask_dir  = os.path.join(seq_dir, "masks")
        print(f"\n{'='*60}\nSequence: {seq_name}")

        # ── 후보 필터링 (시퀀스당 1회) ────────────────────────────────────────
        all_frames = sorted(os.listdir(image_dir))
        candidates = filter_candidates(
            all_frames, image_dir, mask_dir,
            target_val=50, occlude_val=150,
            blur_threshold=args.blur_threshold,
            visible_threshold=args.visible_threshold,
        )
        candidate_names = [c[0] for c in candidates]
        visible_ratios  = np.array([c[1] for c in candidates])
        print(f"  Candidates: {len(candidates)} / {len(all_frames)}")

        if not candidate_names:
            print(f"  [WARN] 후보 없음, 시퀀스 건너뜀")
            run += runs_per_seq
            continue

        # ── DINOv2 피처 추출 (시퀀스당 1회) ──────────────────────────────────
        print("  DINOv2 feature extraction...")
        fps_features_base = extract_features(pipeline, candidate_names, image_dir)

        for n_views in n_views_list:
            if len(candidates) < n_views:
                print(f"  [WARN] n_views={n_views}: 후보 {len(candidates)}개 부족, 건너뜀")
                run += fps_per_nv + rot_per_nv
                continue

            # ── fps 프레임 선택 캐시 (outlier_sigma당 1회) ────────────────────
            fps_cache: dict = {}
            for sigma in ALL_OUTLIER_SIGMAS:
                feats  = fps_features_base.copy()
                cnames = list(candidate_names)
                vrats  = visible_ratios.copy()
                if sigma is not None:
                    feats, cnames, vrats = remove_outlier_views(
                        feats, cnames, vrats, n_views, sigma=sigma)
                if len(cnames) < n_views:
                    print(f"  [WARN] fps n={n_views} sigma={sigma}: "
                          f"아웃라이어 제거 후 {len(cnames)}개, 건너뜀")
                    fps_cache[sigma] = []
                    continue
                idx = select_frames_fps(feats, vrats, cnames, n_views, args.visible_threshold)
                fps_cache[sigma] = [cnames[i] for i in idx]

            # ── rotation 프레임 선택 (rotation 사용 시에만) ──────────────────
            rot_selected = (select_frames_rotation(candidate_names, seq_dir, n_views)
                            if "rotation" in ALL_SELECT_METHODS else [])

            for select_method in ALL_SELECT_METHODS:
                sigma_iter = ALL_OUTLIER_SIGMAS if select_method == "fps" else [None]

                for sigma in sigma_iter:
                    selected_fnames = (fps_cache.get(sigma, [])
                                       if select_method == "fps" else rot_selected)

                    for occlude_method in ALL_OCCLUDE_METHODS:
                        for black_bg in ALL_BLACK_BG:
                            run += 1

                            out_name = f"ch_{select_method}"
                            if occlude_method == "dilate":
                                out_name += f"_dilate{args.dilate_px}"
                            elif occlude_method == "flood":
                                out_name += "_flood"
                            if black_bg:
                                out_name += "_blackbg"
                            if select_method == "fps" and sigma is not None:
                                out_name += f"_os{sigma}"
                            out_dir = os.path.join("./output", seq_name,
                                                   f"n{n_views}", out_name)

                            if os.path.exists(os.path.join(out_dir, "mesh.ply")):
                                skipped += 1
                                print(f"  [{run}/{total}] SKIP  {seq_name}/n{n_views}/{out_name}")
                                continue

                            if not selected_fnames:
                                failed += 1
                                continue

                            print(f"  [{run}/{total}] {seq_name}/n{n_views}/{out_name}")

                            images, masks, used_ids = [], [], []
                            for fname in selected_fnames:
                                fid = os.path.splitext(fname)[0]
                                img, msk = preprocess_frame(
                                    os.path.join(image_dir, fname),
                                    os.path.join(mask_dir, fid + ".png"),
                                    margin=args.margin, output_size=args.output_size,
                                    black_bg=black_bg, occlude_method=occlude_method,
                                    dilate_px=args.dilate_px,
                                )
                                if img is None:
                                    continue
                                images.append(img)
                                masks.append(msk)
                                used_ids.append(fid)

                            if not images:
                                print("    -> 유효 프레임 없음, 건너뜀")
                                failed += 1
                                continue

                            try:
                                _save_outputs(pipeline, images, masks, out_dir,
                                              seq_dir, used_ids, select_method, n_views,
                                              occlude_method, black_bg, sigma, args)
                                print("    -> OK")
                            except Exception as e:
                                print(f"    -> FAILED: {e}")
                                failed += 1

    print("=" * 60)
    print(f"Done.  Total={total}  Skipped={skipped}  Failed={failed}")


if __name__ == "__main__":
# ── 인자 파싱 ────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()

    seq_group = parser.add_mutually_exclusive_group(required=True)
    seq_group.add_argument("--seq_dir", type=str,
                           help="단일 시퀀스 폴더 (single mode)")
    seq_group.add_argument("--train_dir", type=str,
                           help="모든 시퀀스 폴더 (batch mode): 전체 조합 자동 실행")

    parser.add_argument("--margin", type=int, default=40, help="Crop margin around object bbox")
    parser.add_argument("--output_size", type=int, default=512, help="Output image size")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--frames", type=int, nargs="+", help="Explicit frame numbers (single mode only)")
    mode.add_argument("--auto_select", action="store_true", help="Auto keyframe selection (single mode only)")

    parser.add_argument("--select_method", type=str, choices=["fps", "rotation"], default="fps",
                        help="Keyframe selection strategy (single mode only)")
    parser.add_argument("--n_views", type=int, nargs="+", default=None,
                        help="사용할 뷰 수. single mode: 기본 4 / batch mode: 기본 2 4 6 8 10 12. 여러 값 가능 (batch mode)")
    parser.add_argument("--blur_threshold", type=float, default=100.0, help="Laplacian blur threshold")
    parser.add_argument("--visible_threshold", type=float, default=0.4, help="Minimum visible ratio")
    parser.add_argument("--black_bg", action="store_true", help="Black out background pixels (single mode only)")
    parser.add_argument("--no_crop", action="store_true", help="Skip object bbox crop, use full image")
    parser.add_argument("--occlude_method", type=str, choices=["convex", "dilate", "flood"], default="convex",
                        help="Occlusion region method (single mode only)")
    parser.add_argument("--dilate_px", type=int, default=10, help="Dilation radius in pixels (--occlude_method dilate only)")
    parser.add_argument("--outlier_sigma", type=float, nargs="?", const=1.5, default=None,
                        help="Cosine similarity 아웃라이어 제거 강도 (single mode only)")
    args = parser.parse_args()

    # ── 파이프라인 로드 (1회) ─────────────────────────────────────────────────────────

    print("Loading pipeline...")
    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.cuda()

    # ── 실행 분기 ─────────────────────────────────────────────────────────────────────

    if args.train_dir:
        _run_batch(args, pipeline)

    else:
        # ── single mode 검증 ──────────────────────────────────────────────────
        if not args.auto_select and not args.frames:
            parser.error("single mode에서는 --frames 또는 --auto_select 가 필요합니다.")

        n_views = args.n_views[0] if args.n_views else 4

        seq_name = os.path.basename(os.path.normpath(args.seq_dir))
        if args.auto_select:
            output_name = f"ch_{args.select_method}"
        else:
            output_name = f"ch_manual_{len(args.frames)}f"
        if args.occlude_method == "dilate":
            output_name += f"_dilate{args.dilate_px}"
        elif args.occlude_method == "flood":
            output_name += "_flood"
        if args.no_crop:
            output_name += "_nocrop"
        if args.black_bg:
            output_name += "_blackbg"
        if args.auto_select and args.select_method == "fps" and args.outlier_sigma is not None:
            output_name += f"_os{args.outlier_sigma}"
        if args.auto_select:
            output_dir = os.path.join("./output", seq_name, f"n{n_views}", output_name)
        else:
            output_dir = os.path.join("./output", seq_name, output_name)
        os.makedirs(output_dir, exist_ok=True)

        image_dir = os.path.join(args.seq_dir, "rgb")
        mask_dir  = os.path.join(args.seq_dir, "masks")

        # ── 프레임 결정 ──────────────────────────────────────────────────────────────

        if args.auto_select:
            all_frames = sorted(os.listdir(image_dir))
            candidates = filter_candidates(
                all_frames, image_dir, mask_dir,
                target_val=50, occlude_val=150,
                blur_threshold=args.blur_threshold,
                visible_threshold=args.visible_threshold,
            )
            candidate_names = [c[0] for c in candidates]
            visible_ratios = np.array([c[1] for c in candidates])
            print(f"  {len(candidates)} / {len(all_frames)} frames passed filtering")
            assert len(candidates) >= n_views, "후보 프레임이 너무 적습니다. 필터 조건을 낮추세요."

            if args.select_method == "fps":
                features = extract_features(pipeline, candidate_names, image_dir)
                if args.outlier_sigma is not None:
                    features, candidate_names, visible_ratios = remove_outlier_views(
                        features, candidate_names, visible_ratios, n_views, sigma=args.outlier_sigma)
                selected_idx = select_frames_fps(features, visible_ratios, candidate_names,
                                                 n_views, args.visible_threshold)
                selected_fnames = [candidate_names[i] for i in selected_idx]
            else:
                selected_fnames = select_frames_rotation(candidate_names, args.seq_dir, n_views)

            print(f"Selected frames: {selected_fnames}")

            frame_ids  = [os.path.splitext(f)[0] for f in selected_fnames]
            rgb_paths  = [os.path.join(image_dir, f) for f in selected_fnames]
            mask_paths = [os.path.join(mask_dir, os.path.splitext(f)[0] + ".png") for f in selected_fnames]
        else:
            frame_ids  = [f"{f:04d}" for f in args.frames]
            rgb_paths  = [os.path.join(image_dir, f"{f:04d}.jpg") for f in args.frames]
            mask_paths = [os.path.join(mask_dir,  f"{f:04d}.png") for f in args.frames]

        # ── 전처리 ───────────────────────────────────────────────────────────────────

        images, masks, used_ids = [], [], []
        for fid, rp, mp in zip(frame_ids, rgb_paths, mask_paths):
            img, msk = preprocess_frame(rp, mp, margin=args.margin, output_size=args.output_size,
                                        black_bg=args.black_bg, occlude_method=args.occlude_method,
                                        dilate_px=args.dilate_px, no_crop=args.no_crop)
            if img is None:
                print(f"[WARN] frame {fid}: no object pixels, skipping")
                continue
            images.append(img)
            masks.append(msk)
            used_ids.append(fid)

        # ── 파이프라인 실행 + 저장 ────────────────────────────────────────────────────

        _save_outputs(pipeline, images, masks, output_dir,
                      args.seq_dir, used_ids, args.select_method, n_views,
                      args.occlude_method, args.black_bg, args.outlier_sigma, args)
