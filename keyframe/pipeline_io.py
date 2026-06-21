"""후보 필터링 / DINOv2 피처 추출 (B: streaming) / 결과 저장."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Generator

import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm as _tqdm
from tqdm import tqdm

from .decorators import load_mask, timing
# amodal3r / trimesh는 GPU가 필요한 함수 내부에서 lazy import

_tqdm_bar = partial(_tqdm, bar_format="{desc}: {n_fmt}it [{elapsed}, {rate_fmt}]")


# ── 후보 필터링 ───────────────────────────────────────────────────────────────

def _check_frame(fname, image_dir, mask_dir, target_val, occlude_val,
                 blur_threshold, visible_threshold):
    img_bgr = cv2.imread(os.path.join(image_dir, fname))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() <= blur_threshold:
        return None

    mask_arr = load_mask(os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png"))
    obj_pixels  = (mask_arr == target_val).sum()
    hand_pixels = (mask_arr == occlude_val).sum()
    if obj_pixels == 0:
        return None

    obj_mask = mask_arr == target_val
    H, W = obj_mask.shape
    if obj_mask[0,:].any() or obj_mask[H-1,:].any() or obj_mask[:,0].any() or obj_mask[:,W-1].any():
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
        for result in _tqdm_bar(executor.map(check, all_frames),
                                total=len(all_frames), desc="Step 1: Filtering"):
            if result is not None:
                results.append(result)
    results.sort(key=lambda x: x[0])
    return results


# ── B: DINOv2 피처 추출 — streaming generator ─────────────────────────────────

def _image_batch_stream(fnames: list[str], image_dir: str,
                        batch_size: int, n_workers: int) -> Generator:
    """배치 크기만큼만 PIL Image를 메모리에 올리고 즉시 yield."""
    def _load(path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    for i in range(0, len(fnames), batch_size):
        batch = fnames[i:i + batch_size]
        paths = [os.path.join(image_dir, f) for f in batch]
        with ThreadPoolExecutor(max_workers=min(n_workers, len(batch))) as ex:
            yield list(ex.map(_load, paths))


@timing
def extract_features(pipeline, candidate_names, image_dir, n_workers=8, batch_size=16):
    all_features = []
    n_batches = (len(candidate_names) + batch_size - 1) // batch_size

    for batch_imgs in _tqdm_bar(
        _image_batch_stream(candidate_names, image_dir, batch_size, n_workers),
        total=n_batches, desc="Step 2: DINOv2 features (stream)",
    ):
        feats = pipeline.encode_image(batch_imgs)
        all_features.append(feats.mean(dim=1).cpu().numpy())

    return np.concatenate(all_features, axis=0)


# ── 저장 유틸 ─────────────────────────────────────────────────────────────────

def save_grid(pil_list, save_path, ncols=4):
    n = len(pil_list)
    w, h = pil_list[0].size
    nrows = (n + ncols - 1) // ncols
    grid = Image.new("RGB", (ncols * w, nrows * h), (255, 255, 255))
    for i, img in enumerate(pil_list):
        grid.paste(img.convert("RGB"), ((i % ncols) * w, (i // ncols) * h))
    grid.save(save_path)


def extract_glb(gs, mesh, mesh_simplify=0.95, texture_size=1024, export_path="output.glb"):
    from amodal3r.utils import postprocessing_utils
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify,
                                      texture_size=texture_size, verbose=False)
    glb.export(export_path)
    return export_path


def save_mesh(mesh_result, filename):
    import trimesh
    vertices = (mesh_result.vertices.cpu().numpy()
                if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices)
    faces    = (mesh_result.faces.cpu().numpy()
                if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh_result.vertex_attrs is not None:
        attrs = (mesh_result.vertex_attrs.cpu().numpy()
                 if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs)
        mesh.visual.vertex_colors = attrs
    mesh.export(filename)


def save_outputs(pipeline, images, masks, out_dir, seq_dir, used_ids,
                 select_method, n_views, occlude_method, black_bg, outlier_sigma, args):
    from amodal3r.utils import render_utils

    os.makedirs(out_dir, exist_ok=True)
    save_grid(images, os.path.join(out_dir, "input_images.png"))
    save_grid(masks,  os.path.join(out_dir, "input_masks.png"))

    outputs = pipeline.run_multi_image(
        images, masks, seed=1,
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 3},
    )

    video_gs   = render_utils.render_video(outputs['gaussian'][0], bg_color=(1,1,1))['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0],     bg_color=(1,1,1))['normal']
    video = [np.concatenate([a, b], axis=1) for a, b in zip(video_gs, video_mesh)]
    imageio.mimsave(os.path.join(out_dir, "sample_multi.gif"), video, fps=30)

    gaussian = outputs['gaussian'][0]
    mv_gs, _, _ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1,1,1))
    for i, frame in enumerate(mv_gs['color']):
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_gs.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    mesh = outputs['mesh'][0]
    mv_mesh, _, _ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1,1,1))
    for i, frame in enumerate(mv_mesh['normal']):
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_mesh.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    save_mesh(mesh, os.path.join(out_dir, "mesh.ply"))

    config = {
        "seq_dir": seq_dir, "frames": used_ids, "mode": "auto_select",
        "select_method": select_method, "n_views": n_views,
        "occlude_method": occlude_method, "dilate_px": args.dilate_px,
        "no_crop": args.no_crop, "black_bg": black_bg,
        "outlier_sigma": outlier_sigma,
        "blur_threshold": args.blur_threshold,
        "visible_threshold": args.visible_threshold,
        "margin": args.margin, "output_size": args.output_size,
    }
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump(config, f, indent=2)
