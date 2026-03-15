#!/bin/bash
# ASDQ Ablation Study: test different theta1/theta2/ratio combinations per evaluation task
# Usage: cd /path/to/ASDQ && bash run_ablation.sh

set -e

CONFIG="configs/default.yaml"
RESULTS_MD="eval_results/ablation_results.md"
SCALE_PATH="scale_cache/asdq_llava_7b_w4.pt"
LOG_DIR="eval_results/logs"

mkdir -p eval_results/logs scale_cache

# ---------- Evaluation tasks (customize here or via ABLATION_TASKS env) ----------
if [ -n "${ABLATION_TASKS}" ]; then
    IFS=',' read -ra TASKS <<< "$ABLATION_TASKS"
else
    TASKS=( "mmmu_val" "realworldqa" "ocrbench" "ai2d" )
fi

# ---------- Experiment configurations ----------
# Format: "theta1 theta2 ratio"
EXPERIMENTS=(
    "0.0  1.0  0.01"
    "0.1  0.9  0.01"
    "0.3  0.7  0.01"
    "0.5  0.5  0.01"
    "0.7  0.3  0.01"
    "0.9  0.1  0.01"
    "1.0  0.0  0.01"
)

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
NUM_TASKS=${#TASKS[@]}

# Write markdown header
cat > "$RESULTS_MD" << 'HEADER'
# ASDQ Ablation Study Results

Model: llava-onevision-qwen2-7b-ov | W4 quantization | SpQR-style mixed precision

HEADER

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "========================================================"
    echo " Task: ${TASK}"
    echo "========================================================"
    echo ""

    # Section header for this task in results
    echo "" >> "$RESULTS_MD"
    echo "## Task: ${TASK}" >> "$RESULTS_MD"
    echo "" >> "$RESULTS_MD"

    # ---------- FP16 baseline (no quantization, no .pt loaded) ----------
    echo "[Ablation] Running FP16 baseline for ${TASK}..."
    LOG_FP16="${LOG_DIR}/${TASK}_fp16_baseline.log"
    python3 main_eval.py --config "$CONFIG" --tasks "$TASK" --results_md "$RESULTS_MD" --output_path eval_results 2>&1 | tee "$LOG_FP16"
    echo "[Ablation] FP16 baseline for ${TASK} complete."
    echo ""

    # ---------- Ablation experiments for this task ----------
    IDX=0
    for EXP in "${EXPERIMENTS[@]}"; do
        read -r THETA1 THETA2 RATIO <<< "$EXP"
        IDX=$((IDX + 1))
        TAG="t1_${THETA1}_t2_${THETA2}_r_${RATIO}"
        LOG_FILE="${LOG_DIR}/${TASK}_${TAG}.log"

        echo ""
        echo "========================================================"
        echo " Task: ${TASK} | Experiment ${IDX}/${NUM_EXPERIMENTS}: theta1=${THETA1}, theta2=${THETA2}, ratio=${RATIO}"
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
print(f'[Ablation] Updated config: theta1=${THETA1}, theta2=${THETA2}, ratio=${RATIO}')
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
        python3 main_eval.py --config "$CONFIG" --tasks "$TASK" --results_md "$RESULTS_MD" 2>&1 | tee -a "${LOG_FILE}"

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
        echo "[Ablation] Task ${TASK} experiment ${IDX}/${NUM_EXPERIMENTS} complete."
        echo ""
    done

    echo "[Ablation] Task ${TASK} complete (1 FP16 + ${NUM_EXPERIMENTS} ablation runs)."
    echo ""
done

echo "========================================================"
echo " All tasks complete!"
echo " Tasks: ${NUM_TASKS} (${TASKS[*]})"
echo " Per task: 1 FP16 baseline + ${NUM_EXPERIMENTS} ablation experiments"
echo " Results: ${RESULTS_MD}"
echo " Logs:    ${LOG_DIR}/"
echo "========================================================"
