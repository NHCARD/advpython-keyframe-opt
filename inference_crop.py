import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Amodal3R'))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['SPCONV_ALGO'] = 'native'

import argparse
import numpy as np
import imageio
from PIL import Image
from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils
import cv2
import trimesh


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


def preprocess_frame(rgb_path, mask_path, target_val=50, occlude_val=150, margin=40, output_size=512):
    image_np = np.array(Image.open(rgb_path).convert("RGB"))
    mask_np = np.array(Image.open(mask_path))
    H, W = mask_np.shape[:2]

    # amodal mask 생성
    amodal_mask = np.full((H, W), 255, dtype=np.uint8)
    amodal_mask[mask_np == target_val] = 200
    amodal_mask[mask_np == occlude_val] = 0

    # object bbox 기준 square crop
    ys, xs = np.where(mask_np == target_val)
    if len(ys) == 0:
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = max(y1 - y0, x1 - x0) // 2 + margin
    y0c = max(0, cy - half); y1c = min(H, cy + half)
    x0c = max(0, cx - half); x1c = min(W, cx + half)

    image_crop = image_np[y0c:y1c, x0c:x1c]
    mask_crop = amodal_mask[y0c:y1c, x0c:x1c]

    # 정사각형 패딩 → output_size
    size = max(image_crop.shape[:2])
    pad_img = np.ones((size, size, 3), dtype=np.uint8) * 255
    pad_mask = np.ones((size, size), dtype=np.uint8) * 255
    h, w = image_crop.shape[:2]
    oh, ow = (size - h) // 2, (size - w) // 2
    pad_img[oh:oh+h, ow:ow+w] = image_crop
    pad_mask[oh:oh+h, ow:ow+w] = mask_crop

    image_pil = Image.fromarray(pad_img).resize((output_size, output_size), Image.LANCZOS)
    mask_pil = Image.fromarray(pad_mask).resize((output_size, output_size), Image.NEAREST).convert("L")
    return image_pil, mask_pil


pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
pipeline.cuda()

parser = argparse.ArgumentParser()
parser.add_argument("--seq_dir", type=str, required=True, help="Sequence folder absolute path")
parser.add_argument("--frames", type=int, nargs="+", required=True, help="Frame numbers to load")
parser.add_argument("--output_dir", type=str, default="./output/1/")
parser.add_argument("--margin", type=int, default=40, help="Crop margin around object bbox")
parser.add_argument("--output_size", type=int, default=512, help="Output image size")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

images, masks = [], []
for f in args.frames:
    img, msk = preprocess_frame(
        os.path.join(args.seq_dir, f"rgb/{f:04d}.jpg"),
        os.path.join(args.seq_dir, f"masks/{f:04d}.png"),
        margin=args.margin,
        output_size=args.output_size,
    )
    if img is None:
        print(f"[WARN] frame {f:04d}: no object pixels, skipping")
        continue
    images.append(img)
    masks.append(msk)

# save input images and masks as grid (4 columns)
def save_grid(pil_list, save_path, ncols=4):
    n = len(pil_list)
    w, h = pil_list[0].size
    nrows = (n + ncols - 1) // ncols
    grid = Image.new("RGB", (ncols * w, nrows * h), (255, 255, 255))
    for i, img in enumerate(pil_list):
        grid.paste(img.convert("RGB"), ((i % ncols) * w, (i // ncols) * h))
    grid.save(save_path)

save_grid(images, os.path.join(output_dir, "input_images.png"))
save_grid(masks,  os.path.join(output_dir, "input_masks.png"))

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    masks,
    seed=1,
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)

# save as gif
video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
video_mesh = render_utils.render_video(outputs['mesh'][0], bg_color=(1, 1, 1))['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave(os.path.join(output_dir, "sample_multi.gif"), video, fps=30)

# save multi-view gs and mesh
gaussian = outputs['gaussian'][0]
multi_view_gs, _, _ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1))
multi_view_gs = multi_view_gs['color']
for i in range(8):
    output = cv2.cvtColor(multi_view_gs[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{i:03d}_gs.png"), output)

mesh = outputs['mesh'][0]
multi_view_mesh, _, _ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
multi_view_mesh = multi_view_mesh['normal']
for i in range(8):
    output = cv2.cvtColor(multi_view_mesh[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{i:03d}_mesh.png"), output)

save_mesh(mesh, os.path.join(output_dir, "mesh.ply"))

glb_path = os.path.join(output_dir, "mesh.glb")
extract_glb(outputs['gaussian'][0], outputs['mesh'][0], 0.5, 1024, glb_path)

with open(os.path.join(output_dir, "frames.txt"), "w") as f:
    f.write(f"seq_dir: {args.seq_dir}\n")
    f.write(f"frames: {args.frames}\n")
