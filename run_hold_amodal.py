import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Amodal3R'))
os.environ['SPCONV_ALGO'] = 'native'

import numpy as np
import imageio
from PIL import Image
import cv2
import trimesh

from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils


def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs
    mesh.export(filename)


def make_occlude_region(obj_mask: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    """convex hull of visible object mask 안에 있는 손 픽셀만 True로 반환.
    손이 물체를 감싸면 가시 영역이 손 주변을 둘러싸므로 hull이 occluded 영역을 정확히 커버."""
    H, W = obj_mask.shape
    obj_pts = np.argwhere(obj_mask)[:, ::-1].astype(np.int32)  # (col, row)
    if len(obj_pts) < 3:
        return np.zeros((H, W), bool)
    hull      = cv2.convexHull(obj_pts)
    hull_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 1)
    return hand_mask & hull_mask.astype(bool)


SEQ_DIR     = "/home/hsg/hoi_project/MagicHOI/ho3d_v3/train/ShSu12"
RGB_DIR     = os.path.join(SEQ_DIR, "rgb")
MASK_DIR    = os.path.join(SEQ_DIR, "masks")
META_DIR    = os.path.join(SEQ_DIR, "meta")
TARGET_VAL  = 50    # 재구성할 객체
OCCLUDE_VAL = 150   # 가리는 객체 (손)
OUTPUT_SIZE = 512
MARGIN      = 40

TOP_FRAMES  = [520,422,525,1001]   # find_keyframe_shape.py 결과

SEQ_NAME    = os.path.basename(SEQ_DIR)
EXAMPLE_DIR = f"./example/{SEQ_NAME}_poly"
OUTPUT_DIR  = f"./output/{SEQ_NAME}_poly"
os.makedirs(EXAMPLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)


def preprocess_frame(frame_idx):
    image_np = np.array(Image.open(f"{RGB_DIR}/{frame_idx:04d}.jpg").convert("RGB"))
    mask_np  = np.array(Image.open(f"{MASK_DIR}/{frame_idx:04d}.png"))
    H, W     = mask_np.shape[:2]

    # --- Amodal3R 마스크 생성 ---
    obj_mask    = mask_np == TARGET_VAL
    hand_region = mask_np == OCCLUDE_VAL

    amodal_mask = np.full((H, W), 255, dtype=np.uint8)
    amodal_mask[obj_mask] = 200  # visible object

    # visible object를 dilation해서 실제로 object에 닿아있는 손 픽셀만 occluded(0)
    occlude_region = make_occlude_region(obj_mask, hand_region)
    amodal_mask[occlude_region] = 0
    # dilation 밖의 손 픽셀은 255(background) 유지

    # --- object bbox 기준 square crop ---
    ys, xs = np.where(mask_np == TARGET_VAL)
    if len(ys) == 0:
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half   = max(y1 - y0, x1 - x0) // 2 + MARGIN
    y0c = max(0, cy - half); y1c = min(H, cy + half)
    x0c = max(0, cx - half); x1c = min(W, cx + half)

    image_crop = image_np[y0c:y1c, x0c:x1c]
    mask_crop  = amodal_mask[y0c:y1c, x0c:x1c]

    # --- 정사각형 패딩 → 512x512 ---
    size = max(image_crop.shape[:2])
    pad_img  = np.ones((size, size, 3), dtype=np.uint8) * 255
    pad_mask = np.ones((size, size),    dtype=np.uint8) * 255
    h, w = image_crop.shape[:2]
    oh, ow = (size - h) // 2, (size - w) // 2
    pad_img [oh:oh+h, ow:ow+w] = image_crop
    pad_mask[oh:oh+h, ow:ow+w] = mask_crop

    image_pil = Image.fromarray(pad_img).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
    mask_pil  = Image.fromarray(pad_mask).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST).convert("L")

    return image_pil, mask_pil


# --- 전처리 및 example 저장 ---
images, masks = [], []
for i, frame_idx in enumerate(TOP_FRAMES):
    image_pil, mask_pil = preprocess_frame(frame_idx)
    if image_pil is None:
        print(f"[WARN] frame {frame_idx:04d}: no object pixels, skipping")
        continue

    image_pil.save(os.path.join(EXAMPLE_DIR, f"{i:06d}.png"))
    mask_pil.save( os.path.join(EXAMPLE_DIR, f"{i:06d}_mask.png"))
    print(f"[{i}] frame {frame_idx:04d} → {EXAMPLE_DIR}/{i:06d}.png")

    images.append(image_pil)
    masks.append(mask_pil)

print(f"\nTotal {len(images)} images prepared from frames {TOP_FRAMES}")

# --- 파이프라인 ---
pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
pipeline.cuda()

outputs = pipeline.run_multi_image(
    images,
    masks,
    seed=1,
    sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
    slat_sampler_params={"steps": 12, "cfg_strength": 3},
)

# --- GIF ---
video_gs   = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
video_mesh = render_utils.render_video(outputs['mesh'][0],     bg_color=(1, 1, 1))['normal']
video = [np.concatenate([fg, fm], axis=1) for fg, fm in zip(video_gs, video_mesh)]
imageio.mimsave(os.path.join(OUTPUT_DIR, "sample.gif"), video, fps=30)
print("Saved GIF")

# --- Multi-view ---
gaussian = outputs['gaussian'][0]
mv_gs, _, _ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1))
for i, frame in enumerate(mv_gs['color']):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{i:03d}_gs.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

mesh = outputs['mesh'][0]
mv_mesh, _, _ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
for i, frame in enumerate(mv_mesh['normal']):
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{i:03d}_mesh.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

save_mesh(mesh, os.path.join(OUTPUT_DIR, "mesh.ply"))
print("Done. Outputs saved to", OUTPUT_DIR)
