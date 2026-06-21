# advpython-keyframe-opt

HO3D 데이터셋에서 Amodal3R 추론을 위한 **키프레임 선택 파이프라인 최적화** 프로젝트입니다.  
[Amodal3R (ICCV 2025)](https://sm0kywu.github.io/Amodal3R/)를 기반으로, 손-물체 상호작용 영상에서 최적 키프레임을 자동 선택하여 3D 재구성을 수행합니다.

---

## 브랜치 구조 (최적화 전/후)

| 브랜치 | 설명 |
|--------|------|
| `main` | **최적화 전** — `inference_crop_convex.py` (618줄 단일 스크립트) |
| `opt`  | **최적화 후** — `inference_crop_convex_refine_.py` + `keyframe/` 패키지로 분리 |

```
git log --oneline --all
# 2645c09 (opt)  opt
# e664a84 (main) init
```

---

## 최적화 내용 요약

| 항목 | 최적화 전 | 최적화 후 |
|------|-----------|-----------|
| **FPS 알고리즘** | `(N, D)` 임시 배열 반복 생성 | `||a−b||² = ||a||²+||b||²−2aᵀb` 벡터화, BLAS GEMV 1회 |
| **아웃라이어 제거** | `O(N²·D)` 코사인 유사도 행렬 전체 계산 | `O(N·D)` 열 합산 벡터 연산으로 대체 |
| **이미지/마스크 I/O** | 동일 경로 중복 읽기 | `@lru_cache` 로 캐싱, 배치 모드 I/O 최소화 |
| **피처 추출** | 전체 이미지 메모리 적재 후 배치 처리 | 스트리밍 제너레이터(`_image_batch_stream`) 로 배치 단위 메모리 관리 |
| **코드 구조** | 618줄 모놀리식 스크립트 | `keyframe/` 패키지로 역할별 분리 |

### `keyframe/` 패키지 구조

```
keyframe/
├── __init__.py        # 패키지 진입점, sys.path 자동 등록
├── decorators.py      # @timing, @validate_features, LRU 캐시 로더
├── preprocessing.py   # 오클루전 마스크 생성 + crop/pad/resize
├── selection.py       # FPS / Rotation FPS / 아웃라이어 제거
└── pipeline_io.py     # 후보 필터링, DINOv2 피처 추출(스트리밍), 결과 저장
```

---

## 환경 설정

### 1. 저장소 클론 (서브모듈 포함)

```bash
git clone https://github.com/NHCARD/advpython-keyframe-opt.git --recursive
cd advpython-keyframe-opt
```

### 2. conda 환경 생성 및 패키지 설치

```bash
cd Amodal3R
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
conda activate amodal3r
cd ..
```

> **요구 환경**: Ubuntu 22.04 / CUDA 11.8 / PyTorch 2.4.0 / Python 3.10

---

## 데이터 준비

아래 3단계를 순서대로 진행합니다.

### Step 1 — HO3D v3 본체 다운로드

[OneDrive 링크](https://1drv.ms/f/s!AsG9HA3ULXQRlFy5tCZXahAe3bEV?e=BevrKO)에서 **`ho3d_v3`** 폴더를 내려받습니다.

### Step 2 — 세그멘테이션 마스크 추가

같은 OneDrive 링크에서 **`Segmentations_rendered`** 폴더를 내려받아 `ho3d_v3/` 안에 넣습니다.

### Step 3 — 오브젝트 모델 추가

[Google Drive 링크](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view)에서 파일을 내려받아 압축을 풀고, 안에 있는 **`models`** 폴더를 `ho3d_v3/` 안에 넣습니다.

### 최종 폴더 구조

```
ho3d_v3/
├── train/
│   ├── ABF12/
│   │   ├── rgb/         # 0000.jpg, 0001.jpg, ...
│   │   ├── masks/       # 0000.png, 0001.png, ...  (pixel값: 50=물체, 150=손)
│   │   └── meta/        # 0000.pkl, ...
│   ├── BB11/
│   └── ...
├── Segmentations_rendered/
└── models/
```

> **데이터 경로**: 스크립트 기본값은 `--train_root` 인자로 지정합니다.  
> shell 스크립트의 `TRAIN_ROOT` 변수를 실제 `ho3d_v3/train` 경로로 수정하세요.

### 샘플 데이터 (`sample/ABF12`)

저장소에 ABF12 시퀀스의 10프레임(0099 ~ 0999, 100프레임 간격)이 포함되어 있습니다.  
HO3D 전체 데이터 없이도 파이프라인을 즉시 실행해볼 수 있습니다.

```
sample/
└── ABF12/
    ├── rgb/     # 0099.jpg, 0199.jpg, ... 0999.jpg  (10장)
    ├── masks/   # 0099.png, 0199.png, ... 0999.png
    └── meta/    # 0099.pkl, 0199.pkl, ... 0999.pkl
```

---

## 실행 방법

### 최적화 후 코드 실행 (`opt` 브랜치)

```bash
git checkout opt
```

#### 단일 시퀀스 — 수동 프레임 지정

```bash
python inference_crop_convex_refine_.py \
    --seq_dir /path/to/ho3d_v3/train/ABF12 \
    --frames 100 250 400 600
```

#### 단일 시퀀스 — 자동 키프레임 선택 (FPS)

```bash
python inference_crop_convex_refine_.py \
    --seq_dir /path/to/ho3d_v3/train/ABF12 \
    --auto_select \
    --n_views 6 \
    --occlude_method convex \
    --outlier_sigma 1.5
```

#### 전체 데이터셋 배치 실행

```bash
bash run_all_ho3d_convex.sh                         # 기본: n_views=4,6,8
bash run_all_ho3d_convex.sh --n_views 4 6           # 특정 n_views만
```

#### 샘플 데이터로 빠른 테스트 (`sample/ABF12` — 10프레임 수록)

```bash
python inference_crop_convex_refine_.py \
    --seq_dir ./sample/ABF12 \
    --auto_select --n_views 4
```

### 최적화 전 코드 실행 (`main` 브랜치)

```bash
git checkout main
bash run_all_ho3d_convex.sh
```

---

## 성능 측정

`keyframe/decorators.py`의 `@timing` 데코레이터가 주요 함수에 적용되어 있어  
실행 시 아래와 같이 각 단계별 시간이 자동으로 출력됩니다.

```
[timing] extract_features: 12.431s
[timing] _fps_loop: 0.082s
[timing] remove_outlier_views: 0.011s
[timing] select_frames_fps: 0.093s
```

### 알고리즘 복잡도 비교

| 함수 | 최적화 전 | 최적화 후 |
|------|-----------|-----------|
| `_fps_loop` | `O(k · N · D)`, 임시 배열 `(N, D)` 반복 | `O(k · N · D)`, BLAS GEMV — 임시 배열 없음 |
| `remove_outlier_views` | `O(N² · D)` 행렬 | `O(N · D)` 벡터 합산 |
| I/O (배치 모드) | 시퀀스당 중복 읽기 | `lru_cache(maxsize=512)` 캐싱 |

### Synthetic 벤치마크 (`benchmark/run_benchmark.py`)

실제 실험 조건(`D=384, K=32`)으로 before / after를 자동 측정합니다.

```bash
python benchmark/run_benchmark.py
```

**고정 조건**

| 항목 | 값 |
|------|----|
| D (임베딩 차원) | 384 (DINOv2 ViT-S/14) |
| K (선택 키프레임 수) | 32 |
| N (프레임 수) | 500 / 1000 / 2000 / 4000 / 8000 |
| 반복 횟수 R | N≤4000: 10회, N=8000: 5회 |
| seed | 42 (재현 가능) |
| warm-up | 1회 |

**측정 항목**: `perf_counter` 실행 시간(mean ± std), `tracemalloc` peak 메모리(MB)

**ablation**: before → A만 적용 → A+B → A+B+D 누적 비교로 개념별 기여도 분리

**산출물**

```
benchmark/results/
├── benchmark_results.csv   # N × version별 시간·메모리 수치
├── env_info.txt            # 측정 환경 (OS, Python, numpy, CPU)
├── time_mem_loglog.png     # 시간·메모리·log-log 그래프 (복잡도 차수 실측)
└── ablation.png            # ablation 비교 그래프
```

**주요 결과 (Intel Xeon Gold 5220R, numpy 2.2.5)**

| N | fps 속도향상 | outlier 속도향상 | outlier 메모리 절감 |
|---|------------|----------------|-------------------|
| 500 | 6.3× | 3.9× | 5.9 → 1.5 MB |
| 2000 | 13.6× | 8.1× | 74.7 → 5.9 MB |
| 8000 | 24.6× | 50.8× | 1122 → 23.8 MB |

log-log 기울기: `outlier_before=2.008` → `outlier_after=0.985` (O(N²) → O(N) 실측 확인)

### 수동 벤치마크

실제 파이프라인과 동일한 조건(`D=384, K=32`)으로 빠르게 확인할 수 있습니다.

```bash
python - <<'EOF'
import time, numpy as np
from keyframe.selection import _fps_loop, remove_outlier_views

N, D, K = 1000, 384, 32   # D=384: DINOv2 ViT-S/14, K=32: 선택 키프레임 수
feats = np.random.default_rng(42).standard_normal((N, D)).astype(np.float64)
feats /= np.linalg.norm(feats, axis=1, keepdims=True)
vrats = np.random.default_rng(42).uniform(0.3, 1.0, N)
names = [str(i) for i in range(N)]

t0 = time.perf_counter()
for _ in range(10):
    _fps_loop(feats, K)
print(f"FPS ×10: {time.perf_counter()-t0:.3f}s")

t0 = time.perf_counter()
for _ in range(10):
    remove_outlier_views(feats, names, vrats, K)
print(f"Outlier removal ×10: {time.perf_counter()-t0:.3f}s")
EOF
```

---

## 주요 스크립트 목록

| 파일 | 설명 |
|------|------|
| `inference_crop_convex_refine_.py` | **최적화 후** 메인 스크립트 |
| `inference_crop_convex.py` | **최적화 전** 원본 스크립트 (main 브랜치) |
| `inference_crop.py` | 단순 crop 버전 (수동 프레임) |
| `inference_ho3d.py` | FPS 기반 시퀀스별 실행 |
| `inference_ho3d_auto.py` | 적응형 임계값 자동 필터링 버전 |
| `inference_ho3d_kmens.py` | K-Means 기반 키프레임 선택 버전 |
| `run_hold_amodal.py` | 단일 시퀀스 수동 실행 |
| `run_all_ho3d_convex.sh` | 전체 배치 실행 래퍼 |
| `synthetic_data.py` | 더미 데이터 생성 |
| `benchmark/run_benchmark.py` | Synthetic 벤치마크 (before/after 자동 측정) |
| `benchmark/results/` | 벤치마크 산출물 (CSV, 그래프) |
