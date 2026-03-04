# W8A8 INT8 真量化 PPL 评测工具

在 omni-infer v1.0.0 (昇腾 910) 上对任意模型进行 FP16 baseline + W8A8 INT8 真量化 WikiText-2 PPL 评测。

**评测方式**: 先用 `vllm serve` 拉起模型服务，再通过 OpenAI API 调用 lm-eval 评测 PPL。

## 文件清单

```
├── setup.sh                     一键脚本 (安装/量化/serve/评测)
├── quantize_w8a8_cpu.py         W8A8 CPU 量化脚本
├── wikitext_local.yaml          lm-eval 任务配置
├── preprocess_wikitext_local.py word_perplexity 计算逻辑
└── data/
    └── wikitext2_doc_level/     WikiText-2 测试数据 (7MB)
```

## 使用方法

### 1. 启动 omni-infer v1.0.0 容器

```bash
docker run -d --name int8-eval \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /path/to/your/models:/models:ro \
  -v /path/to/this/repo:/data/int8-ppl-eval \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  --entrypoint /bin/sh \
  swr.cn-east-4.myhuaweicloud.com/omni-ci/omniinfer-a2-arm:v1.0.0-vllm \
  -c 'while true; do sleep 3600; done'
```

> **不要**挂载到 `/workspace`，会遮挡 omni-npu 源码。

### 2. 进入容器，一键执行

```bash
docker exec -it int8-eval bash
bash /data/int8-ppl-eval/setup.sh all /models/YourModel
```

这会依次执行:
1. **install** — 安装 lm-eval、llmcompressor，修复环境
2. **quantize** — W8A8 量化 (CPU, data-free)
3. **eval_fp16** — 启动 FP16 vllm serve → API 评测 PPL → 停止 serve
4. **eval_w8a8** — 启动 W8A8 vllm serve → API 评测 PPL → 停止 serve

### 3. 分步执行

```bash
bash setup.sh install                      # 安装依赖
bash setup.sh quantize  /models/YourModel  # W8A8 量化
bash setup.sh eval_fp16 /models/YourModel  # FP16 PPL (自动 serve/stop)
bash setup.sh eval_w8a8 /models/YourModel  # W8A8 PPL (自动 serve/stop)
```

也可以手动控制 serve 生命周期:
```bash
bash setup.sh serve /models/YourModel      # 手动启动 serve
bash setup.sh eval  /models/YourModel      # 手动评测
bash setup.sh stop                         # 手动停止 serve
```

### 4. 大模型多卡 / 自定义插件

```bash
# 92B 模型 4 卡并行
TP_SIZE=4 bash setup.sh all /models/Large-92B

# 指定 vllm 插件
VLLM_PLUGINS="omni-npu,omni_custom_models,omni_npu_patches" \
  bash setup.sh eval_fp16 /models/MyModel

# 自定义端口
SERVE_PORT=8080 bash setup.sh eval_fp16 /models/MyModel

# 额外 vllm serve 参数
EXTRA_SERVE_ARGS="--max-model-len 4096" bash setup.sh eval_fp16 /models/MyModel
```

## 已知问题

| 问题 | 原因 | setup.sh 处理方式 |
|------|------|------------------|
| llmcompressor 降级 torch 到 2.8.0 | pip 依赖解析 | 自动备份+恢复 aarch64 wheel |
| `forward_oot() got unexpected keyword 'is_prefill'` | omni-npu 兼容性 | 自动 patch |
| `OMNI_NPU_PATCHES_DIR` 解析错误 | sys.argv 解析失败 | 自动从 config.json 检测 model_type |
| graph capture `stream is captured` 错误 | CANN 不兼容 | enforce_eager=True |
| engine 关闭 core dump | vllm 已知问题 | 结果不受影响 |

## 参考结果 (Qwen3-0.6B, omni-infer v1.0.0)

| 模型 | word_perplexity | 劣化 |
|------|----------------|------|
| FP16 (API 方式) | 60.0541 | — |
| W8A8 (API 方式) | 61.2216 | +1.94% |
