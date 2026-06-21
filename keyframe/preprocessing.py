"""오클루전 마스크 생성 + 프레임 전처리 (crop / pad / resize)."""
import numpy as np
import cv2
from PIL import Image

from .decorators import load_image, load_mask


def make_occlude_region(obj_mask: np.ndarray, hand_mask: np.ndarray,
                        method: str = "convex", dilate_px: int = 10) -> np.ndarray:
    if method == "dilate":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        region = cv2.dilate(obj_mask.astype(np.uint8), kernel).astype(bool)
        return hand_mask & region

    elif method == "flood":
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


def preprocess_frame(rgb_path, mask_path, target_val=50, occlude_val=150,
                     margin=40, output_size=512, black_bg=False,
                     occlude_method="convex", dilate_px=10, no_crop=False):
    image_np = load_image(rgb_path).copy()
    mask_np  = load_mask(mask_path)
    H, W = mask_np.shape[:2]

    obj_mask  = mask_np == target_val
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
        image_region, mask_region = image_np, amodal_mask
    else:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = max(y1 - y0, x1 - x0) // 2 + margin
        y0c = max(0, cy - half); y1c = min(H, cy + half)
        x0c = max(0, cx - half); x1c = min(W, cx + half)
        image_region = image_np[y0c:y1c, x0c:x1c]
        mask_region  = amodal_mask[y0c:y1c, x0c:x1c]

    size = max(image_region.shape[:2])
    img_bg = 0 if black_bg else 255
    pad_img  = np.full((size, size, 3), img_bg, dtype=np.uint8)
    pad_mask = np.ones((size, size), dtype=np.uint8) * 255
    h, w = image_region.shape[:2]
    oh, ow = (size - h) // 2, (size - w) // 2
    pad_img[oh:oh+h, ow:ow+w]  = image_region
    pad_mask[oh:oh+h, ow:ow+w] = mask_region

    image_pil = Image.fromarray(pad_img).resize((output_size, output_size), Image.LANCZOS)
    mask_pil  = Image.fromarray(pad_mask).resize((output_size, output_size), Image.NEAREST).convert("L")
    return image_pil, mask_pil
