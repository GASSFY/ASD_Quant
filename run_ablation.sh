#!/bin/bash
# ASDQ Ablation Study: test different theta1/theta2/ratio combinations
# Usage: cd /path/to/ASD_Quant && bash run_ablation.sh

set -e

CONFIG="configs/default.yaml"
RESULTS_MD="eval_results/ablation_results.md"
SCALE_PATH="scale_cache/asdq_llava_7b_w4.pt"
LOG_DIR="eval_results/logs"

mkdir -p eval_results/logs scale_cache

# Write markdown header
cat > "$RESULTS_MD" << 'HEADER'
# ASDQ Ablation Study Results

Model: llava-onevision-qwen2-7b-ov | W4 quantization | SpQR-style mixed precision

HEADER

# ---------- Experiment configurations ----------
# Format: "theta1 theta2 ratio"
EXPERIMENTS=(
    "1.0  0.0  0.01"
    "0.5  0.5  0.01"
    "1.0  0.0  0.1"
    "0.5  0.5  0.1"
)

TOTAL=${#EXPERIMENTS[@]}
IDX=0

for EXP in "${EXPERIMENTS[@]}"; do
    read -r THETA1 THETA2 RATIO <<< "$EXP"
    IDX=$((IDX + 1))
    TAG="t1_${THETA1}_t2_${THETA2}_r_${RATIO}"
    LOG_FILE="${LOG_DIR}/${TAG}.log"

    echo ""
    echo "========================================================"
    echo " Experiment ${IDX}/${TOTAL}: theta1=${THETA1}, theta2=${THETA2}, ratio=${RATIO}"
    echo "========================================================"
    echo ""

    # Step 1: Modify default.yaml
    python3 -c "
import yaml
with open('${CONFIG}', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['asd_theta1'] = ${THETA1}
cfg['asd_theta2'] = ${THETA2}
cfg['asd_high_precision_ratio'] = ${RATIO}
with open('${CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(f'[Ablation] Updated config: theta1={${THETA1}}, theta2={${THETA2}}, ratio={${RATIO}}')
"

    # Step 2: Quantize (with timing)
    echo "[Ablation] Running quantization..."
    QUANT_START=$(date +%s)
    python3 main_quant.py --config "$CONFIG" 2>&1 | tee "${LOG_FILE}"
    QUANT_END=$(date +%s)
    QUANT_TIME=$((QUANT_END - QUANT_START))
    QUANT_MIN=$((QUANT_TIME / 60))
    QUANT_SEC=$((QUANT_TIME % 60))
    echo "[Ablation] Quantization finished in ${QUANT_MIN}m ${QUANT_SEC}s (${QUANT_TIME}s total)"

    # Step 3: Evaluate (results saved to markdown + console log)
    echo "[Ablation] Running evaluation..."
    python3 main_eval.py --config "$CONFIG" --results_md "$RESULTS_MD" 2>&1 | tee -a "${LOG_FILE}"

    # Step 4: Append quantization time to markdown results
    echo "" >> "$RESULTS_MD"
    echo "> Quantization time: **${QUANT_MIN}m ${QUANT_SEC}s** (${QUANT_TIME}s)" >> "$RESULTS_MD"
    echo "" >> "$RESULTS_MD"

    # Step 5: Delete .pt file to save disk space
    if [ -f "$SCALE_PATH" ]; then
        rm -f "$SCALE_PATH"
        echo "[Ablation] Deleted ${SCALE_PATH} to save disk space."
    fi

    echo ""
    echo "[Ablation] Experiment ${IDX}/${TOTAL} complete."
    echo ""
done

echo "========================================================"
echo " All ${TOTAL} experiments complete!"
echo " Results: ${RESULTS_MD}"
echo " Logs:    ${LOG_DIR}/"
echo "========================================================"
