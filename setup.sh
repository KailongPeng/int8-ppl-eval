#!/bin/bash
# ============================================================
# omni-infer v1.0.0 W8A8 INT8 PPL 评测全流程
#
# 方式 B: 先 vllm serve 拉起模型，再通过 API 评测 PPL
#
# 用法:
#   bash setup.sh install                           — 安装依赖 + 修复环境
#   bash setup.sh quantize  /models/YourModel       — W8A8 量化 (CPU)
#   bash setup.sh serve     /models/YourModel       — 启动 vllm serve
#   bash setup.sh eval      /models/YourModel       — 通过 API 评测 PPL
#   bash setup.sh stop                              — 停止 vllm serve
#   bash setup.sh eval_fp16 /models/YourModel       — serve + eval + stop (FP16)
#   bash setup.sh eval_w8a8 /models/YourModel       — serve + eval + stop (W8A8)
#   bash setup.sh all       /models/YourModel       — 全部依次执行
#
# 环境变量 (可选):
#   VLLM_PLUGINS    — vllm 插件列表 (默认: 自动检测)
#   SERVE_PORT      — vllm serve 端口 (默认: 8000)
#   TP_SIZE         — tensor parallel size (默认: 1)
#   EXTRA_SERVE_ARGS — 额外的 vllm serve 参数
# ============================================================

set -uo pipefail

TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVE_PORT="${SERVE_PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"

# 确保本地请求不走代理
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1"
export NO_PROXY="${no_proxy}"

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
import json, os
cfg = os.path.join('$1', 'config.json')
with open(cfg) as f:
    print(json.load(f).get('model_type', 'auto').lower())
" 2>/dev/null || echo "auto"
}

# 修正 yaml 中的数据路径 (只执行一次)
fix_yaml_path() {
    if grep -q '__TASK_DIR__' "${TASK_DIR}/wikitext_local.yaml"; then
        sed -i "s|__TASK_DIR__|${TASK_DIR}|g" "${TASK_DIR}/wikitext_local.yaml"
        info "Fixed dataset path → ${TASK_DIR}/data/wikitext2_doc_level"
    fi
}

# 等待 vllm serve 就绪
wait_for_serve() {
    info "Waiting for vllm serve to be ready on port ${SERVE_PORT} ..."
    for i in $(seq 1 120); do
        if curl --noproxy '*' -s "http://7.150.11.4:8000/v1/models" > /dev/null 2>&1; then
            info "vllm serve is ready!"
            return 0
        fi
        sleep 2
    done
    error "vllm serve failed to start within 240 seconds. Check logs."
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
    info "Installing lm-eval[api], llmcompressor, ray ..."
    pip install "lm-eval[api]" llmcompressor==0.8.1 ray 2>&1 | tail -5

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

# ---- serve ----
do_serve() {
    [ -z "${1:-}" ] && error "Usage: setup.sh serve /path/to/model"
    local model_path="$1"

    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

    local model_type
    model_type=$(detect_model_type "${model_path}")

    # 设置 VLLM_PLUGINS (用户可通过环境变量覆盖)
    export OMNI_NPU_PATCHES_DIR="${model_type}"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
    export VLLM_USE_V1="${VLLM_USE_V1:-0}"

    # 检测服务是否已在运行
    if curl --noproxy '*' -s "http://7.150.11.4:8000/v1/models" > /dev/null 2>&1; then
        info "Service already running on port ${SERVE_PORT}, skipping launch"
        return 0
    fi

    info "=== Starting vllm serve ==="
    info "Model: ${model_path}"
    info "Model type: ${model_type}"
    info "Port: ${SERVE_PORT}"
    info "TP size: ${TP_SIZE}"
    [ -n "${VLLM_PLUGINS:-}" ] && info "VLLM_PLUGINS: ${VLLM_PLUGINS}"

    # 启动服务
    if [ "${model_type}" = "pangu_v2_moe" ]; then
        info "Using run_pangu.sh for pangu_v2_moe model"
        bash /home/p00929643/omni-npu/start_server/run_pangu.sh
        wait_for_serve
    else
        vllm serve "${model_path}" \
            --dtype auto \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --tensor-parallel-size "${TP_SIZE}" \
            --host 0.0.0.0 \
            --port "${SERVE_PORT}" \
            ${EXTRA_SERVE_ARGS:-} \
            > "${TASK_DIR}/vllm_serve.log" 2>&1 &

        SERVE_PID=$!
        echo "${SERVE_PID}" > "${TASK_DIR}/.serve_pid"
        info "Server started (PID: ${SERVE_PID})"
        wait_for_serve
    fi
}

# ---- stop ----
do_stop() {
    info "=== Stopping vllm serve ==="
    if [ -f "${TASK_DIR}/.serve_pid" ]; then
        local pid
        pid=$(cat "${TASK_DIR}/.serve_pid")
        kill "${pid}" 2>/dev/null || true
        sleep 2
        kill -9 "${pid}" 2>/dev/null || true
        rm -f "${TASK_DIR}/.serve_pid"
    fi
    # 兜底: 杀掉所有 vllm 相关进程
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "APIServer" 2>/dev/null || true
    sleep 1
    info "vllm serve stopped"
}

# ---- eval (通过 API) ----
do_eval() {
    [ -z "${1:-}" ] && error "Usage: setup.sh eval /path/to/model [output_dir]"
    local model_path="$1"
    local output_dir="${2:-$(dirname "${TASK_DIR}")/results/$(basename "${model_path}")}"

    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    fix_yaml_path

    info "=== PPL Eval via API ==="
    info "Model: ${model_path}"
    info "API: http://7.150.11.4:8000/v1/completions"
    info "Output: ${output_dir}"

    mkdir -p "${output_dir}"
    HF_DATASETS_OFFLINE=1 \
    lm_eval --model local-completions \
        --model_args "model=${model_path},base_url=http://7.150.11.4:8000/v1/completions,tokenizer_backend=huggingface,tokenizer=${model_path},trust_remote_code=True" \
        --include_path "${TASK_DIR}" \
        --tasks wikitext_local \
        --batch_size auto \
        --output_path "${output_dir}"

    info "Eval done → ${output_dir}"
}

# ---- eval_fp16: serve → eval → stop ----
do_eval_fp16() {
    [ -z "${1:-}" ] && error "Usage: setup.sh eval_fp16 /path/to/model"
    resolve_paths "$1"
    info "====== FP16 Pipeline: serve → eval → stop ======"
    do_serve "${MODEL_PATH}"
    do_eval "${MODEL_PATH}" "${RESULTS_FP16}" || true
    do_stop
}

# ---- eval_w8a8: serve → eval → stop ----
do_eval_w8a8() {
    [ -z "${1:-}" ] && error "Usage: setup.sh eval_w8a8 /path/to/model"
    resolve_paths "$1"
    [ -d "${QUANT_OUTPUT}" ] || error "Quantized model not found: ${QUANT_OUTPUT}. Run 'setup.sh quantize' first."
    info "====== W8A8 Pipeline: serve → eval → stop ======"
    do_serve "${QUANT_OUTPUT}"
    do_eval "${QUANT_OUTPUT}" "${RESULTS_W8A8}" || true
    do_stop
}

# ---- main ----
case "${1:-help}" in
    install)   do_install ;;
    quantize)  do_quantize "${2:-}" ;;
    serve)     do_serve "${2:-}" ;;
    eval)      do_eval "${2:-}" "${3:-}" ;;
    stop)      do_stop ;;
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
        echo "  serve     /path/to/model   Start vllm serve"
        echo "  eval      /path/to/model   Eval PPL via API (serve must be running)"
        echo "  stop                       Stop vllm serve"
        echo "  eval_fp16 /path/to/model   FP16: serve + eval + stop"
        echo "  eval_w8a8 /path/to/model   W8A8: serve + eval + stop"
        echo "  all       /path/to/model   Run all steps"
        echo ""
        echo "Environment variables:"
        echo "  VLLM_PLUGINS         vllm plugins (e.g. 'omni-npu,omni_custom_models')"
        echo "  SERVE_PORT           vllm serve port (default: 8000)"
        echo "  TP_SIZE              tensor parallel size (default: 1)"
        echo "  EXTRA_SERVE_ARGS     extra args for vllm serve"
        echo ""
        echo "Examples:"
        echo "  bash setup.sh all /models/Qwen3-0.6B"
        echo "  TP_SIZE=4 bash setup.sh all /models/Large-92B"
        echo "  VLLM_PLUGINS='omni-npu,omni_custom_models' bash setup.sh eval_fp16 /models/MyModel"
        ;;
esac
