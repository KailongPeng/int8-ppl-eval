#!/bin/bash
# ============================================================
# omni-infer v1.0.0 W8A8 INT8 PPL 评测全流程
#
# 用法:
#   bash setup.sh install                          — 安装依赖 + 修复环境
#   bash setup.sh quantize  /models/YourModel      — W8A8 量化
#   bash setup.sh eval_fp16 /models/YourModel      — FP16 baseline PPL
#   bash setup.sh eval_w8a8 /models/YourModel      — W8A8 INT8 PPL
#   bash setup.sh all       /models/YourModel      — 全部依次执行
# ============================================================

set -uo pipefail

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# 从模型路径推导量化输出路径和结果路径
resolve_paths() {
    MODEL_PATH="$1"
    MODEL_NAME="$(basename "${MODEL_PATH}")"
    QUANT_OUTPUT="$(dirname "${MODEL_PATH}")/${MODEL_NAME}-W8A8"
    RESULTS_DIR="$(dirname "${TASK_DIR}")/results"
    RESULTS_FP16="${RESULTS_DIR}/${MODEL_NAME}-fp16"
    RESULTS_W8A8="${RESULTS_DIR}/${MODEL_NAME}-w8a8"
}

# 自动检测模型类型 → omni-npu patches 目录名
detect_model_type() {
    python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('$1')
print(c.model_type.lower())
" 2>/dev/null || echo "auto"
}

# 修正 yaml 中的数据路径 (只执行一次)
fix_yaml_path() {
    if grep -q '__TASK_DIR__' "${TASK_DIR}/wikitext_local.yaml"; then
        sed -i "s|__TASK_DIR__|${TASK_DIR}|g" "${TASK_DIR}/wikitext_local.yaml"
        info "Fixed dataset path → ${TASK_DIR}/data/wikitext2_doc_level"
    fi
}

# ---- install ----
do_install() {
    info "=== Install dependencies ==="

    # Source CANN
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

    # 备份 torch wheel
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    info "Current torch: ${TORCH_VER}"
    WHEEL_DIR="${TASK_DIR}/.torch_wheel"
    mkdir -p "${WHEEL_DIR}"
    if [ ! -f "${WHEEL_DIR}"/torch-*.whl ]; then
        info "Saving torch wheel for restoration ..."
        pip download "torch==${TORCH_VER}" --no-deps -d "${WHEEL_DIR}" 2>/dev/null || true
    fi

    # 安装
    info "Installing lm-eval, llmcompressor, ray ..."
    pip install lm-eval llmcompressor==0.8.1 ray 2>&1 | tail -5

    # 修复 torch
    NEW_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ "${NEW_VER}" != "${TORCH_VER}" ]; then
        warn "torch changed: ${TORCH_VER} → ${NEW_VER}"
        WHEEL=$(ls "${WHEEL_DIR}"/torch-*.whl 2>/dev/null | head -1)
        if [ -n "${WHEEL}" ]; then
            info "Restoring torch from: ${WHEEL}"
            pip install "${WHEEL}" --force-reinstall --no-deps 2>&1 | tail -3
        else
            error "No saved wheel. Manually reinstall torch ${TORCH_VER} (aarch64 wheel, NOT PyPI CPU index)"
        fi
    fi

    # Pin versions
    pip install "compressed-tensors==0.12.2" --no-deps 2>&1 | tail -2
    # transformers: 保持 >=4.57 即可，不强制 pin 具体版本
    TRANS_VER=$(python3 -c "import transformers; v=transformers.__version__; print(v)")
    info "transformers: ${TRANS_VER}"

    # 修复 yaml
    fix_yaml_path

    # Patch rotary embedding
    ROTARY_FILE=$(find /workspace -path "*/omni_npu/layers/rotary_embedding/rotary_embedding_torch_npu.py" 2>/dev/null | head -1)
    if [ -n "${ROTARY_FILE}" ] && ! grep -q '\*\*kwargs' "${ROTARY_FILE}"; then
        info "Patching rotary_embedding forward_oot ..."
        sed -i 's/key: torch.Tensor | None = None,$/key: torch.Tensor | None = None,\n        **kwargs,/' "${ROTARY_FILE}"
    fi

    # 验证
    info "=== Versions ==="
    python3 -c "
import torch; print(f'  torch:              {torch.__version__}')
try:
    import torch_npu; print(f'  torch_npu:          {torch_npu.__version__}')
except: print('  torch_npu:          N/A')
import vllm; print(f'  vllm:               {vllm.__version__}')
import compressed_tensors; print(f'  compressed-tensors: {compressed_tensors.__version__}')
import transformers; print(f'  transformers:       {transformers.__version__}')
import llmcompressor; print(f'  llmcompressor:      {llmcompressor.__version__}')
import lm_eval; print(f'  lm-eval:            {lm_eval.__version__}')
"
    info "Install done!"
}

# ---- quantize ----
do_quantize() {
    [ -z "${1:-}" ] && error "Usage: setup.sh quantize /path/to/model"
    resolve_paths "$1"
    info "=== W8A8 Quantize (CPU) ==="
    info "Input:  ${MODEL_PATH}"
    info "Output: ${QUANT_OUTPUT}"

    if [ -f "${QUANT_OUTPUT}/model.safetensors" ]; then
        warn "Already exists: ${QUANT_OUTPUT}, skipping"
        return 0
    fi

    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python3 "${TASK_DIR}/quantize_w8a8_cpu.py" \
        --model "${MODEL_PATH}" --output "${QUANT_OUTPUT}"
    info "Quantize done → ${QUANT_OUTPUT}"
}

# ---- eval ----
do_eval() {
    local label="$1" model_path="$2" output_dir="$3"
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    fix_yaml_path

    local model_type
    model_type=$(detect_model_type "${model_path}")
    info "=== ${label} PPL Eval ==="
    info "Model: ${model_path}"
    info "Patches: ${model_type}"

    mkdir -p "${output_dir}"
    OMNI_NPU_PATCHES_DIR="${model_type}" \
    ASCEND_RT_VISIBLE_DEVICES=0 \
    VLLM_USE_V1=0 \
    HF_DATASETS_OFFLINE=1 \
    lm_eval --model vllm \
        --model_args "pretrained=${model_path},dtype=auto,gpu_memory_utilization=0.8,enforce_eager=True" \
        --include_path "${TASK_DIR}" \
        --tasks wikitext_local \
        --batch_size auto \
        --output_path "${output_dir}" \
    || warn "Engine exited with error (results likely still valid)"

    info "${label} done → ${output_dir}"
}

do_eval_fp16() {
    [ -z "${1:-}" ] && error "Usage: setup.sh eval_fp16 /path/to/model"
    resolve_paths "$1"
    do_eval "FP16" "${MODEL_PATH}" "${RESULTS_FP16}"
}

do_eval_w8a8() {
    [ -z "${1:-}" ] && error "Usage: setup.sh eval_w8a8 /path/to/model"
    resolve_paths "$1"
    [ -f "${QUANT_OUTPUT}/model.safetensors" ] || error "Quantized model not found: ${QUANT_OUTPUT}. Run 'setup.sh quantize' first."
    do_eval "W8A8" "${QUANT_OUTPUT}" "${RESULTS_W8A8}"
}

# ---- main ----
case "${1:-help}" in
    install)   do_install ;;
    quantize)  do_quantize "${2:-}" ;;
    eval_fp16) do_eval_fp16 "${2:-}" ;;
    eval_w8a8) do_eval_w8a8 "${2:-}" ;;
    all)
        [ -z "${2:-}" ] && error "Usage: setup.sh all /path/to/model"
        do_install
        do_quantize "$2"
        do_eval_fp16 "$2"
        do_eval_w8a8 "$2"
        resolve_paths "$2"
        info "========================================="
        info "All done! Results:"
        info "  FP16: ${RESULTS_FP16}"
        info "  W8A8: ${RESULTS_W8A8}"
        info "========================================="
        ;;
    *)
        echo "Usage: bash setup.sh <command> [model_path]"
        echo ""
        echo "Commands:"
        echo "  install                    Install deps + fix environment"
        echo "  quantize  /path/to/model   W8A8 quantization (CPU)"
        echo "  eval_fp16 /path/to/model   FP16 baseline PPL"
        echo "  eval_w8a8 /path/to/model   W8A8 INT8 PPL"
        echo "  all       /path/to/model   Run all steps"
        echo ""
        echo "Example:"
        echo "  bash setup.sh all /models/MyPrivateModel"
        ;;
esac
