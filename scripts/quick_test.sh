#!/usr/bin/env bash
set -euo pipefail

# Order: InternVL2-26B -> InternVL2-8B -> LLaVA-1.5-7B -> InternVL2-1B -> InternVL2-2B (single GPU)
# run_one_model "internvl2_26b" "internvl2" "pretrained=OpenGVLab/InternVL2-26B" "configs/internvl2/Eval/eval.yaml"
# run_one_model "internvl2_8b" "internvl2" "pretrained=OpenGVLab/InternVL2-8B" "configs/internvl2/Eval/eval.yaml"
# run_one_model "llava_v15_7b" "llava" "pretrained=liuhaotian/llava-v1.5-7b" "configs/llava_v15/Eval/eval.yaml"
# run_one_model "internvl2_1b" "internvl2" "pretrained=OpenGVLab/InternVL2-1B" "configs/internvl2/Eval/eval.yaml"
# run_one_model "internvl2_2b" "internvl2" "pretrained=OpenGVLab/InternVL2-2B" "configs/internvl2/Eval/eval.yaml"


cd ..   # 移动到量化目录
# ===== 你只需要改这两处模型参数 =====
MODEL="internvl2"   # 可选: llava / internvl2 / llava_onevision
MODEL_ARGS="pretrained=OpenGVLab/InternVL2-1B"

# ===== 固定数据路径（按你的要求）=====
DATA_PATH="/root/autodl-tmp/hf_home/datasets/coco/sharegpt4v_coco_only.json"
IMAGE_FOLDER="/root/autodl-tmp/hf_home/datasets"

# ===== 运行参数（最小烟测）=====
N_SAMPLES=8
W_BIT=4
W_GROUP=128
OUT_DIR="quick_smoke_${MODEL}"
SCALE_PATH="${OUT_DIR}/asdq_w4.pt"
RESULTS_MD="${OUT_DIR}/smoke_results.md"

mkdir -p "${OUT_DIR}"

echo "[1/2] 量化烟测开始..."
python main_quant.py \
  --model "${MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --batch_size 1 \
  --calib_data coco \
  --n_samples "${N_SAMPLES}" \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --run_process \
  --pseudo_quant \
  --w_bit "${W_BIT}" \
  --w_group "${W_GROUP}" \
  --asd_mixed_precision \
  --asd_theta1 0.8 \
  --asd_theta2 0.2 \
  --asd_high_precision_ratio 0.01 \
  --asd_low_w_bit 4 \
  --scale_path "${SCALE_PATH}"

echo "[2/2] 评估烟测开始..."
python main_eval.py \
  --model "${MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --batch_size 1 \
  --tasks "mmmu_val" \
  --limit 2 \
  --scale_path "${SCALE_PATH}" \
  --output_path "${OUT_DIR}" \
  --results_md "${RESULTS_MD}"

echo "完成。输出目录: ${OUT_DIR}"
echo "量化权重: ${SCALE_PATH}"
echo "评估结果: ${RESULTS_MD}"