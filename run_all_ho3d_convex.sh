#!/bin/bash
# inference_crop_convex.py 배치 모드 실행 래퍼
# 모든 조합(select_method, occlude_method, black_bg, outlier_sigma) × n_views를
# 파이프라인 1회 로드로 효율적으로 실행합니다.
#
# 사용법:
#   bash run_all_ho3d_convex.sh                    # 기본: n_views=4,6,8
#   bash run_all_ho3d_convex.sh --n_views 4 6      # 특정 n_views만

cd "$(dirname "$0")"

python inference_crop_convex.py \
    --train_dir /home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train \
    "$@"
