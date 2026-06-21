#!/usr/bin/env python3
"""
Benchmark: inference_crop_convex.py (before) vs inference_crop_convex_refine_.py (after)

측정 대상: HO3D 키프레임 선택 파이프라인 핵심 연산
  - FPS 프레임 선택 (A: 벡터화 최적화)
  - 아웃라이어 제거 (A: O(N²·D) → O(N·D))
  - 이미지 적재 (B: list 전체 materialize → streaming generator)

Synthetic data: seed 기반 np.random.randn(N, D=384) 정규화 임베딩
               visible_ratio: np.random.uniform(0.3, 1.0)
Fixed: K=32, D=384, SEED=42, warm-up=1
"""

import csv
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np

# ─── 측정 환경 ────────────────────────────────────────────────────────────────
ENV = {
    "os": platform.platform(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "cpu": platform.processor() or "unknown",
    "D": 384,
    "K": 32,
    "seed": 42,
    "warmup": 1,
}

try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if "model name" in line:
                ENV["cpu"] = line.split(":", 1)[1].strip()
                break
except Exception:
    pass

# ─── 설정 ────────────────────────────────────────────────────────────────────
N_VALUES   = [500, 1000, 2000, 4000, 8000]
REPEATS    = {500: 10, 1000: 10, 2000: 10, 4000: 10, 8000: 5}
K          = ENV["K"]
D          = ENV["D"]
SEED       = ENV["seed"]
BATCH_SIZE = 16   # B 개선 시뮬레이션용 배치 크기


# ─── Synthetic data ──────────────────────────────────────────────────────────

def make_features(N: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    feats = rng.standard_normal((N, D)).astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    visible = rng.uniform(0.3, 1.0, size=N).astype(np.float64)
    return feats, visible


# ─── BEFORE functions (inference_crop_convex.py) ─────────────────────────────

def fps_before(features: np.ndarray, k: int) -> list[int]:
    """원본 FPS: 매 반복마다 (N,D) 임시 배열 생성."""
    centroid = features.mean(axis=0)
    anchor = int(np.sum((features - centroid) ** 2, axis=1).argmax())
    selected_idx = [anchor]
    dists = np.full(len(features), np.inf)
    for _ in range(k - 1):
        d = np.sum((features - features[selected_idx[-1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        dists[selected_idx] = -np.inf
        selected_idx.append(int(np.argmax(dists)))
    return selected_idx


def outlier_before(features: np.ndarray, k: int, sigma: float = 1.5) -> np.ndarray:
    """원본 아웃라이어 제거: O(N²·D) 행렬 곱셈."""
    N = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    feats_norm = features / norms
    sim_matrix = feats_norm @ feats_norm.T   # (N, N)  ← 병목
    np.fill_diagonal(sim_matrix, np.nan)
    mean_sim = np.nanmean(sim_matrix, axis=1)
    threshold = mean_sim.mean() - sigma * mean_sim.std()
    keep = mean_sim >= threshold
    if keep.sum() < k:
        keep = np.zeros(N, bool)
        keep[np.argsort(mean_sim)[-k:]] = True
    return np.where(keep)[0]


def load_all_before(N: int) -> list:
    """원본: N개 이미지를 list로 모두 메모리에 적재."""
    rng = np.random.default_rng(SEED)
    # 224×224×3 uint8 배열로 PIL Image 로드를 시뮬레이션
    return [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(N)]


# ─── AFTER functions (inference_crop_convex_refine_.py) ──────────────────────

def fps_after(features: np.ndarray, k: int) -> list[int]:
    """개선 FPS: 제곱 norm 사전 계산 + BLAS GEMV → (N,D) 임시 배열 제거.

    ||a-b||² = ||a||² + ||b||² - 2·aᵀb
    precomputed norms_sq: O(N·D) 1회, 반복당 GEMV O(N·D) (temp 없음)
    """
    N = len(features)
    norms_sq = np.einsum('nd,nd->n', features, features)   # (N,)
    centroid = features.mean(axis=0)
    d0 = norms_sq + float(np.dot(centroid, centroid)) - 2.0 * (features @ centroid)
    np.maximum(d0, 0.0, out=d0)
    anchor = int(d0.argmax())

    selected_idx = [anchor]
    dists = np.full(N, np.inf)

    for _ in range(k - 1):
        last = selected_idx[-1]
        d = norms_sq + norms_sq[last] - 2.0 * (features @ features[last])
        np.maximum(d, 0.0, out=d)
        np.minimum(dists, d, out=dists)
        dists[selected_idx] = -np.inf
        selected_idx.append(int(np.argmax(dists)))

    return selected_idx


def outlier_after(features: np.ndarray, k: int, sigma: float = 1.5) -> np.ndarray:
    """개선 아웃라이어 제거: O(N·D) — N×N 행렬 불필요.

    mean_sim[i] = (feats_norm[i] · col_sum − self_sim[i]) / (N−1)
    col_sum = Σⱼ feats_norm[j]  →  두 번의 O(N·D) 연산으로 완료
    """
    N = len(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    feats_norm = features / norms

    col_sum  = feats_norm.sum(axis=0)                           # (D,)
    row_dot  = feats_norm @ col_sum                             # (N,)
    self_sim = np.einsum('nd,nd->n', feats_norm, feats_norm)   # ≈ 1.0
    mean_sim = (row_dot - self_sim) / max(N - 1, 1)

    threshold = mean_sim.mean() - sigma * mean_sim.std()
    keep = mean_sim >= threshold
    if keep.sum() < k:
        keep = np.zeros(N, bool)
        keep[np.argsort(mean_sim)[-k:]] = True
    return np.where(keep)[0]


def load_stream_after(N: int) -> None:
    """개선: 배치 단위 generator — 최대 BATCH_SIZE개 이미지만 동시 메모리 점유."""
    rng = np.random.default_rng(SEED)
    for i in range(0, N, BATCH_SIZE):
        bsz = min(BATCH_SIZE, N - i)
        batch = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(bsz)]
        # 배치 처리 시뮬레이션 (DINOv2 인코딩 대신 평균 계산)
        _ = np.stack(batch).mean()
        # 배치 변수 해제 → 다음 배치가 같은 메모리 재사용
        del batch


# ─── 결과 동일성 검증 ─────────────────────────────────────────────────────────

def check_equivalence():
    print("=" * 60)
    print("결과 동일성 검증 (N=300, K=16)")
    rng = np.random.default_rng(SEED)
    feats = rng.standard_normal((300, D))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8

    # FPS: 동일한 인덱스 집합 기대
    set_b = set(fps_before(feats, 16))
    set_a = set(fps_after(feats, 16))
    fps_match = set_b == set_a
    print(f"  FPS 인덱스 완전 일치: {'✓' if fps_match else '✗'}")
    if not fps_match:
        print(f"    불일치 인덱스: {set_b ^ set_a}")

    # 아웃라이어: 부동소수점 차이 허용 (95% 이상 overlap)
    keep_b = set(outlier_before(feats, 16).tolist())
    keep_a = set(outlier_after(feats, 16).tolist())
    if keep_b:
        overlap = len(keep_b & keep_a) / len(keep_b)
        print(f"  아웃라이어 제거 overlap: {overlap*100:.1f}% (≥95% 기준)")
        assert overlap >= 0.95, f"overlap {overlap:.2f} < 0.95"
    print("  검증 완료\n")


# ─── 측정 ────────────────────────────────────────────────────────────────────

def measure(fn: Callable, repeats: int) -> dict:
    """warm-up 1회 후 repeats회 측정. perf_counter + tracemalloc peak."""
    fn()  # warm-up

    times, peaks = [], []
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peaks.append(peak / 1024 / 1024)  # MB

    return {
        "mean_time": float(np.mean(times)),
        "std_time":  float(np.std(times)),
        "mean_mem":  float(np.mean(peaks)),
        "std_mem":   float(np.std(peaks)),
    }


# ─── 개별 함수 벤치마크 ────────────────────────────────────────────────────────

def bench_individual(N: int, feats: np.ndarray, R: int) -> list[dict]:
    rows = []
    configs = [
        ("fps_before",      lambda: fps_before(feats, K)),
        ("fps_after",       lambda: fps_after(feats, K)),
        ("outlier_before",  lambda: outlier_before(feats, K)),
        ("outlier_after",   lambda: outlier_after(feats, K)),
        ("load_before",     lambda: load_all_before(N)),
        ("load_after",      lambda: load_stream_after(N)),
    ]
    for name, fn in configs:
        s = measure(fn, R)
        rows.append({"N": N, "version": name, **s})
        print(f"    {name:18s}  {s['mean_time']*1000:8.2f}±{s['std_time']*1000:.2f} ms  "
              f"mem {s['mean_mem']:6.1f}±{s['std_mem']:.1f} MB")
    return rows


# ─── 누적 ablation 벤치마크 ──────────────────────────────────────────────────

def bench_ablation(N: int, feats: np.ndarray, R: int) -> list[dict]:
    """before / A / A+B / A+B+D 누적 비교.

    A+B+D는 A+B와 런타임이 거의 같음 (데코레이터 오버헤드 < 1ms).
    별도 측정으로 오버헤드를 정량화.
    """
    import functools

    # 데코레이터 오버헤드 측정용 wrapper
    _timing_overhead = []
    def timed_wrapper(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            _timing_overhead.append(time.perf_counter() - t0)
            return r
        return wrapper

    before_fn  = lambda: (outlier_before(feats, K), fps_before(feats, K), load_all_before(N))
    a_fn       = lambda: (outlier_after(feats, K),  fps_after(feats, K),  load_all_before(N))
    a_b_fn     = lambda: (outlier_after(feats, K),  fps_after(feats, K),  load_stream_after(N))
    # A+B+D: A+B에 timing decorator 추가 시뮬레이션
    a_b_d_fn   = lambda: (
        timed_wrapper(outlier_after)(feats, K),
        timed_wrapper(fps_after)(feats, K),
        load_stream_after(N),
    )

    rows = []
    for name, fn in [
        ("before",  before_fn),
        ("A_only",  a_fn),
        ("A+B",     a_b_fn),
        ("A+B+D",   a_b_d_fn),
    ]:
        s = measure(fn, R)
        rows.append({"N": N, "version": name, **s})
        print(f"    {name:8s}  {s['mean_time']*1000:8.2f}±{s['std_time']*1000:.2f} ms  "
              f"mem {s['mean_mem']:6.1f}±{s['std_mem']:.1f} MB")
    return rows


# ─── log-log 기울기 (복잡도 차수 실측) ───────────────────────────────────────

def loglog_slope(ns: list[int], times: list[float]) -> float:
    """log(time) ~ slope * log(N) 선형 회귀로 복잡도 차수 추정."""
    log_n = np.log(ns)
    log_t = np.log(times)
    slope, _ = np.polyfit(log_n, log_t, 1)
    return float(slope)


# ─── 그래프 ──────────────────────────────────────────────────────────────────

def plot_results(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib 없음, 그래프 생략")
        return

    # ── time vs N ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 개별 함수 시간 비교
    ax = axes[0]
    pairs = [("fps_before", "fps_after"), ("outlier_before", "outlier_after"), ("load_before", "load_after")]
    colors = [("steelblue", "tomato"), ("forestgreen", "darkorange"), ("mediumpurple", "gold")]
    for (b_key, a_key), (cb, ca) in zip(pairs, colors):
        ns_b = [r["N"] for r in rows if r["version"] == b_key]
        ts_b = [r["mean_time"] * 1000 for r in rows if r["version"] == b_key]
        ns_a = [r["N"] for r in rows if r["version"] == a_key]
        ts_a = [r["mean_time"] * 1000 for r in rows if r["version"] == a_key]
        label = b_key.replace("_before", "")
        ax.plot(ns_b, ts_b, 'o--', color=cb, label=f"{label} (before)")
        ax.plot(ns_a, ts_a, 's-',  color=ca, label=f"{label} (after)")
    ax.set_xlabel("N (frames)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Execution Time: Before vs After (per function)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── memory vs N ───────────────────────────────────────────────
    ax = axes[1]
    for (b_key, a_key), (cb, ca) in zip(pairs, colors):
        ns_b = [r["N"] for r in rows if r["version"] == b_key]
        ms_b = [r["mean_mem"] for r in rows if r["version"] == b_key]
        ns_a = [r["N"] for r in rows if r["version"] == a_key]
        ms_a = [r["mean_mem"] for r in rows if r["version"] == a_key]
        label = b_key.replace("_before", "")
        ax.plot(ns_b, ms_b, 'o--', color=cb, label=f"{label} (before)")
        ax.plot(ns_a, ms_a, 's-',  color=ca, label=f"{label} (after)")
    ax.set_xlabel("N (frames)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory: Before vs After (per function)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── log-log (복잡도 차수) ──────────────────────────────────────
    ax = axes[2]
    for key, color, ls in [
        ("fps_before",     "steelblue",   "--"),
        ("fps_after",      "tomato",      "-"),
        ("outlier_before", "forestgreen", "--"),
        ("outlier_after",  "darkorange",  "-"),
    ]:
        ns = [r["N"] for r in rows if r["version"] == key]
        ts = [r["mean_time"] for r in rows if r["version"] == key]
        if len(ns) < 2:
            continue
        slope = loglog_slope(ns, ts)
        ax.plot(np.log(ns), np.log([t * 1000 for t in ts]),
                marker='o', linestyle=ls, color=color,
                label=f"{key} (slope={slope:.2f})")
    ax.set_xlabel("log(N)")
    ax.set_ylabel("log(Time, ms)")
    ax.set_title("Log-Log Plot (복잡도 차수 실측)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "time_mem_loglog.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  그래프 저장: {path}")

    # ── ablation plot ──────────────────────────────────────────────
    ablation_versions = ["before", "A_only", "A+B", "A+B+D"]
    ablation_rows = [r for r in rows if r["version"] in ablation_versions]
    if ablation_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors_abl = {"before": "steelblue", "A_only": "tomato", "A+B": "forestgreen", "A+B+D": "darkorange"}

        for ax_i, metric, ylabel in [(0, "mean_time", "Time (ms)"), (1, "mean_mem", "Peak Memory (MB)")]:
            ax = axes[ax_i]
            for ver in ablation_versions:
                ns = [r["N"] for r in ablation_rows if r["version"] == ver]
                vs = [r[metric] * (1000 if metric == "mean_time" else 1) for r in ablation_rows if r["version"] == ver]
                if ns:
                    ax.plot(ns, vs, 'o-', label=ver, color=colors_abl[ver])
            ax.set_xlabel("N (frames)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Ablation: {ylabel}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = out_dir / "ablation.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Ablation 그래프 저장: {path}")


# ─── 복잡도 요약 출력 ─────────────────────────────────────────────────────────

def print_complexity_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("복잡도 차수 실측 (log-log 기울기)")
    for key in ["fps_before", "fps_after", "outlier_before", "outlier_after"]:
        ns = [r["N"] for r in rows if r["version"] == key]
        ts = [r["mean_time"] for r in rows if r["version"] == key]
        if len(ns) >= 2:
            slope = loglog_slope(ns, ts)
            print(f"  {key:20s}: slope = {slope:.3f}  (이론: fps≈1.0, outlier_before≈2.0, outlier_after≈1.0)")


def print_speedup_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("속도 향상 비율 (before / after)")
    print(f"{'N':>6}  {'fps speedup':>12}  {'outlier speedup':>16}  {'load speedup':>13}")
    for N in N_VALUES:
        def get(ver):
            for r in rows:
                if r["N"] == N and r["version"] == ver:
                    return r["mean_time"]
            return None
        fps_b = get("fps_before");   fps_a = get("fps_after")
        out_b = get("outlier_before"); out_a = get("outlier_after")
        lod_b = get("load_before");  lod_a = get("load_after")
        fps_sp  = fps_b / fps_a   if fps_a and fps_a > 0  else float('nan')
        out_sp  = out_b / out_a   if out_a and out_a > 0  else float('nan')
        lod_sp  = lod_b / lod_a   if lod_a and lod_a > 0 else float('nan')
        print(f"  {N:>5}  {fps_sp:>11.2f}x  {out_sp:>15.2f}x  {lod_sp:>12.2f}x")


# ─── 1. mean ± std 표 출력 ───────────────────────────────────────────────────

def print_std_table(rows: list[dict]) -> None:
    """과제 요구: 평균과 표준편차를 'X.XX ± Y.YY ms' 형태로 출력."""
    print("\n" + "=" * 60)
    print("실측값 표 (mean ± std)")
    print(f"{'N':>6}  {'버전':22}  {'시간 (ms)':>18}  {'메모리 (MB)':>18}  {'CV(%)':>7}")
    print("-" * 80)
    targets = ["fps_before", "fps_after", "outlier_before", "outlier_after",
               "load_before", "load_after"]
    for N in N_VALUES:
        for ver in targets:
            r = next((x for x in rows if x["N"] == N and x["version"] == ver), None)
            if r is None:
                continue
            t_mean = r["mean_time"] * 1000
            t_std  = r["std_time"]  * 1000
            cv     = (t_std / t_mean * 100) if t_mean > 0 else 0
            print(f"  {N:>5}  {ver:22}  "
                  f"{t_mean:7.2f} ± {t_std:5.2f} ms  "
                  f"{r['mean_mem']:6.1f} ± {r['std_mem']:4.1f} MB  "
                  f"{cv:6.1f}%")
        print()


# ─── 2. B: 실제 파일 I/O 측정 ────────────────────────────────────────────────

def _setup_test_files(N: int, tmpdir: Path, img_size: int = 224) -> list[str]:
    """N개 JPEG 이미지를 tmpdir에 생성하고 경로 목록 반환."""
    from PIL import Image
    rng = np.random.default_rng(SEED)
    paths = []
    for i in range(N):
        arr = rng.integers(60, 200, (img_size, img_size, 3), dtype=np.uint8)
        p = tmpdir / f"{i:04d}.jpg"
        Image.fromarray(arr).save(p, quality=85)
        paths.append(str(p))
    return paths


def bench_real_io(out_dir: Path, N: int = 500, R: int = 5) -> dict:
    """B 개선: 실제 JPEG 파일 I/O로 일괄 적재 vs 스트리밍 측정.

    시뮬레이션과 달리 실제 디스크 읽기 포함.
    목적: 시간은 동일, peak 메모리만 감소함을 실측으로 확인.
    """
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    import tempfile, shutil

    tmpdir = Path(tempfile.mkdtemp())
    try:
        print(f"\n  [B 실제 I/O] N={N}개 테스트 이미지 생성 중...")
        paths = _setup_test_files(N, tmpdir)

        def _load(p: str) -> Image.Image:
            return Image.open(p).convert('RGB')

        def load_all():
            with ThreadPoolExecutor(max_workers=8) as ex:
                imgs = list(ex.map(_load, paths))
            return imgs

        def load_stream():
            for i in range(0, len(paths), BATCH_SIZE):
                batch_p = paths[i:i + BATCH_SIZE]
                with ThreadPoolExecutor(max_workers=min(8, len(batch_p))) as ex:
                    batch = list(ex.map(_load, batch_p))
                del batch

        s_all    = measure(load_all,    R)
        s_stream = measure(load_stream, R)

        result = {
            "N": N,
            "load_all_ms":    s_all["mean_time"]    * 1000,
            "load_all_std":   s_all["std_time"]     * 1000,
            "load_all_mem":   s_all["mean_mem"],
            "stream_ms":      s_stream["mean_time"] * 1000,
            "stream_std":     s_stream["std_time"]  * 1000,
            "stream_mem":     s_stream["mean_mem"],
            "time_ratio":     s_stream["mean_time"] / s_all["mean_time"],
            "mem_reduction":  (1 - s_stream["mean_mem"] / s_all["mean_mem"]) * 100,
        }

        print(f"  결과 (N={N}, R={R}, 실제 JPEG I/O):")
        print(f"    일괄 적재:  {result['load_all_ms']:.1f} ± {result['load_all_std']:.1f} ms  "
              f"peak {result['load_all_mem']:.1f} MB")
        print(f"    스트리밍:   {result['stream_ms']:.1f} ± {result['stream_std']:.1f} ms  "
              f"peak {result['stream_mem']:.1f} MB")
        print(f"    시간 비율:  {result['time_ratio']:.2f}x  (1.0 = 동일)")
        print(f"    메모리 절감: {result['mem_reduction']:.1f}%")

        # CSV 추가 저장
        csv_path = out_dir / "real_io_results.csv"
        with open(csv_path, "w", newline="") as f:
            import csv as _csv
            writer = _csv.DictWriter(f, fieldnames=list(result.keys()))
            writer.writeheader()
            writer.writerow(result)
        print(f"    저장: {csv_path}")

        return result
    finally:
        shutil.rmtree(tmpdir)


# ─── 3. D: lru_cache hit/miss 정량 측정 ──────────────────────────────────────

def bench_lru_cache(out_dir: Path, N: int = 500, R: int = 5) -> dict:
    """D 개선: lru_cache 효과 정량화.

    배치 모드 패턴 시뮬레이션:
      Phase 1 (필터링): 각 마스크를 1회 로드     → N misses
      Phase 2 (전처리): 동일 마스크를 다시 로드  → N hits (캐시 ON)
                                                  → N disk reads (캐시 OFF)

    측정: Phase 2 시간 및 전체 hit/miss 카운트
    """
    from functools import lru_cache as _lru_cache
    from PIL import Image
    import tempfile, shutil

    tmpdir = Path(tempfile.mkdtemp())
    try:
        print(f"\n  [D lru_cache] N={N}개 테스트 마스크 생성 중...")
        # 마스크: 단채널 PNG (실제 HO3D 형식과 동일)
        rng = np.random.default_rng(SEED)
        paths = []
        for i in range(N):
            arr = rng.integers(0, 3, (480, 640), dtype=np.uint8) * 50  # 0, 50, 100
            p = tmpdir / f"{i:04d}.png"
            Image.fromarray(arr).save(p)
            paths.append(str(p))

        @_lru_cache(maxsize=512)
        def cached_load(path: str) -> np.ndarray:
            arr = np.array(Image.open(path))
            arr.flags.writeable = False
            return arr

        def uncached_load(path: str) -> np.ndarray:
            return np.array(Image.open(path))

        # ── 캐시 ON: phase1(miss) + phase2(hit) ──────────────────────
        def with_cache():
            cached_load.cache_clear()
            for p in paths:           # phase1: N misses
                cached_load(p)
            for p in paths:           # phase2: N hits
                cached_load(p)

        # ── 캐시 OFF: phase1 + phase2 모두 disk read ──────────────────
        def without_cache():
            for p in paths:           # phase1
                uncached_load(p)
            for p in paths:           # phase2
                uncached_load(p)

        # phase2만 따로 측정 (hit vs miss 직접 비교)
        cached_load.cache_clear()
        for p in paths:
            cached_load(p)            # 캐시 워밍

        def phase2_hit():
            for p in paths:
                cached_load(p)

        def phase2_miss():
            for p in paths:
                uncached_load(p)

        s_total_with    = measure(with_cache,    R)
        s_total_without = measure(without_cache, R)
        s_phase2_hit    = measure(phase2_hit,    R)
        s_phase2_miss   = measure(phase2_miss,   R)

        # 최종 캐시 통계
        cached_load.cache_clear()
        for p in paths:
            cached_load(p)
        for p in paths:
            cached_load(p)
        info = cached_load.cache_info()

        result = {
            "N": N,
            "total_with_cache_ms":    s_total_with["mean_time"]    * 1000,
            "total_without_cache_ms": s_total_without["mean_time"] * 1000,
            "phase2_hit_ms":          s_phase2_hit["mean_time"]    * 1000,
            "phase2_miss_ms":         s_phase2_miss["mean_time"]   * 1000,
            "phase2_speedup":         s_phase2_miss["mean_time"] / s_phase2_hit["mean_time"],
            "total_speedup":          s_total_without["mean_time"] / s_total_with["mean_time"],
            "cache_hits":             info.hits,
            "cache_misses":           info.misses,
            "hit_rate_pct":           info.hits / max(info.hits + info.misses, 1) * 100,
        }

        print(f"  결과 (N={N}, R={R}, 실제 PNG I/O):")
        print(f"    전체 (phase1+2) with cache:    {result['total_with_cache_ms']:.1f} ms")
        print(f"    전체 (phase1+2) without cache: {result['total_without_cache_ms']:.1f} ms  "
              f"→ {result['total_speedup']:.2f}x 느림")
        print(f"    Phase 2 hit  (캐시 적중):  {result['phase2_hit_ms']:.2f} ms")
        print(f"    Phase 2 miss (디스크 읽기): {result['phase2_miss_ms']:.2f} ms  "
              f"→ {result['phase2_speedup']:.1f}x 차이")
        print(f"    cache_info: hits={result['cache_hits']}, "
              f"misses={result['cache_misses']}, "
              f"hit_rate={result['hit_rate_pct']:.1f}%")

        csv_path = out_dir / "lru_cache_results.csv"
        with open(csv_path, "w", newline="") as f:
            import csv as _csv
            writer = _csv.DictWriter(f, fieldnames=list(result.keys()))
            writer.writeheader()
            writer.writerow(result)
        print(f"    저장: {csv_path}")

        return result
    finally:
        shutil.rmtree(tmpdir)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("측정 환경")
    for k, v in ENV.items():
        print(f"  {k}: {v}")

    check_equivalence()

    all_rows = []

    for N in N_VALUES:
        R = REPEATS[N]
        feats, _ = make_features(N)
        print(f"\n{'='*60}")
        print(f"N = {N:5d}  R = {R}  (warm-up 1회 포함)")

        print("  [개별 함수]")
        all_rows.extend(bench_individual(N, feats, R))

        print("  [누적 ablation: before → A → A+B → A+B+D]")
        all_rows.extend(bench_ablation(N, feats, R))

    # CSV 저장
    csv_path = out_dir / "benchmark_results.csv"
    fieldnames = ["N", "version", "mean_time", "std_time", "mean_mem", "std_mem"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV 저장: {csv_path}")

    # 환경 정보 저장
    env_path = out_dir / "env_info.txt"
    with open(env_path, "w") as f:
        for k, v in ENV.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nN_values: {N_VALUES}\n")
        f.write(f"K={K}, D={D}, seed={SEED}, batch_size={BATCH_SIZE}\n")
        f.write(f"repeats: {REPEATS}\n")
    print(f"환경 정보 저장: {env_path}")

    print_complexity_summary(all_rows)
    print_speedup_summary(all_rows)
    print_std_table(all_rows)

    print("\n[추가 실험 1] B: 실제 파일 I/O 측정")
    bench_real_io(out_dir, N=500, R=5)

    print("\n[추가 실험 2] D: lru_cache hit/miss 정량 측정")
    bench_lru_cache(out_dir, N=500, R=5)

    print("\n그래프 생성 중...")
    plot_results(all_rows, out_dir)

    print("\n완료.")


if __name__ == "__main__":
    main()
