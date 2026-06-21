#!/bin/bash

SEQ_DIR=/home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train/SM2
FRAMES="146 216 573 658"

# 1) 크롭 있음 + 흰 배경
python inference_crop_convex.py \
    --seq_dir $SEQ_DIR \
    --frames $FRAMES

# 2) 크롭 있음 + 검정 배경
python inference_crop_convex.py \
    --seq_dir $SEQ_DIR \
    --frames $FRAMES \
    --black_bg

# 3) 크롭 없음 + 흰 배경
python inference_crop_convex.py \
    --seq_dir $SEQ_DIR \
    --frames $FRAMES \
    --no_crop

# 4) 크롭 없음 + 검정 배경
python inference_crop_convex.py \
    --seq_dir $SEQ_DIR \
    --frames $FRAMES \
    --no_crop \
    --black_bg
