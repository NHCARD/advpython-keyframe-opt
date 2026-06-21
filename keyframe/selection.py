"""A: 키프레임 선택 — FPS (벡터화) / 아웃라이어 제거 (O(N·D)) / Rotation FPS."""
import pickle

import numpy as np
import cv2
from tqdm import tqdm

from .decorators import timing, validate_features


@timing
@validate_features
def _fps_loop(features: np.ndarray, k: int) -> list[int]:
    """FPS 핵심 루프: 제곱 norm 사전 계산으로 (N,D) 임시 배열 제거.

    ||a-b||² = ||a||² + ||b||² - 2·aᵀb
    반복당 BLAS GEMV 1회 → 임시 배열 없이 O(N·D) 처리.
    """
    N = len(features)
    norms_sq = np.einsum('nd,nd->n', features, features)

    centroid = features.mean(axis=0)
    d0 = norms_sq + float(np.dot(centroid, centroid)) - 2.0 * (features @ centroid)
    np.maximum(d0, 0.0, out=d0)
    anchor = int(d0.argmax())

    selected = [anchor]
    dists = np.full(N, np.inf)

    for _ in range(k - 1):
        last = selected[-1]
        d = norms_sq + norms_sq[last] - 2.0 * (features @ features[last])
        np.maximum(d, 0.0, out=d)
        np.minimum(dists, d, out=dists)
        dists[selected] = -np.inf
        selected.append(int(np.argmax(dists)))

    return selected


@timing
@validate_features
def remove_outlier_views(features: np.ndarray, candidate_names: list,
                         visible_ratios: np.ndarray, k: int,
                         sigma: float = 1.5) -> tuple:
    """아웃라이어 제거: O(N²·D) 행렬 → O(N·D) 열 합산 벡터.

    mean_sim[i] = (feats_norm[i]·col_sum − self_sim[i]) / (N−1)
    N×N 행렬 없이 두 번의 O(N·D) 연산으로 완료.
    """
    if sigma <= 0 or len(candidate_names) <= k:
        return features, candidate_names, visible_ratios

    N = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    feats_norm = features / norms

    col_sum  = feats_norm.sum(axis=0)
    row_dot  = feats_norm @ col_sum
    self_sim = np.einsum('nd,nd->n', feats_norm, feats_norm)
    mean_sim = (row_dot - self_sim) / max(N - 1, 1)

    threshold = mean_sim.mean() - sigma * mean_sim.std()
    keep = mean_sim >= threshold
    if keep.sum() < k:
        keep = np.zeros(N, bool)
        keep[np.argsort(mean_sim)[-k:]] = True

    removed = [(candidate_names[i], float(mean_sim[i])) for i in range(N) if not keep[i]]
    if removed:
        for name, sim in removed:
            print(f"  [아웃라이어 제거] {name} (mean_sim={sim:.3f}, threshold={threshold:.3f})")
    else:
        print(f"  [아웃라이어 없음] mean_sim 범위: {mean_sim.min():.3f}~{mean_sim.max():.3f}")

    keep_idx = np.where(keep)[0]
    return features[keep_idx], [candidate_names[i] for i in keep_idx], visible_ratios[keep_idx]


@timing
def select_frames_fps(features, visible_ratios, candidate_names, k, visible_threshold):
    """FPS 후 visible ratio 기준 교체."""
    selected_idx = _fps_loop(features, k)

    not_selected = sorted(
        [i for i in range(len(features)) if i not in selected_idx],
        key=lambda i: -visible_ratios[i],
    )
    for pos, idx in enumerate(selected_idx):
        if visible_ratios[idx] < visible_threshold and not_selected:
            rep = not_selected.pop(0)
            print(f"  [교체] {candidate_names[idx]} (vis={visible_ratios[idx]:.3f})"
                  f" → {candidate_names[rep]} (vis={visible_ratios[rep]:.3f})")
            selected_idx[pos] = rep

    return sorted(selected_idx)


@timing
def select_frames_rotation(candidate_names, seq_dir, k):
    """meta pkl의 objRot을 9D flatten해서 FPS로 자세 다양성 최대화."""
    import os
    meta_dir = os.path.join(seq_dir, "meta")
    valid_names, features = [], []

    for fname in tqdm(candidate_names, desc="Step 2: Loading rotations"):
        stem = os.path.splitext(fname)[0]
        with open(os.path.join(meta_dir, stem + ".pkl"), 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        obj_rot = meta.get('objRot')
        if obj_rot is None:
            continue
        R = cv2.Rodrigues(obj_rot)[0] if obj_rot.shape != (3, 3) else obj_rot
        valid_names.append(fname)
        features.append(R.flatten())

    selected_idx = _fps_loop(np.array(features), k)
    return sorted([valid_names[i] for i in selected_idx])
