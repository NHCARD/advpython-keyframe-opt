#!/bin/bash
set -e

TRAIN_ROOT="/home/inuri64/PycharmProjects/Amodal3R/ho3d_v3/train"
OUTPUT_ROOT="./output"
SCRIPT="inference_ho3d.py"
LOG_DIR="./logs_ho3d"

mkdir -p "$LOG_DIR"

SEQUENCES=(ABF12 BB11 GPMF11 GSF14 MC1 MDF10 ShSu10 SMu40)

echo "Running ${#SEQUENCES[@]} sequences: ${SEQUENCES[*]}"
echo "========================================"

for SEQ in "${SEQUENCES[@]}"; do
    OUTPUT_DIR="$OUTPUT_ROOT/hold_${SEQ}_ho3d"

    echo "[START] $SEQ  ($(date '+%H:%M:%S'))"
    python "$SCRIPT" \
        --seq "$SEQ" \
        --train_root "$TRAIN_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        2>&1 | tee "$LOG_DIR/${SEQ}.log"

    if [ $? -eq 0 ]; then
        echo "[DONE]  $SEQ"
    else
        echo "[FAIL]  $SEQ — see $LOG_DIR/${SEQ}.log"
    fi
    echo "----------------------------------------"
done

echo "All sequences finished."
