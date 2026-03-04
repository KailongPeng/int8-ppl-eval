# W8A8 INT8 真量化 PPL 评测工具

在 omni-infer v1.0.0 (昇腾 910) 上对任意模型进行 FP16 baseline + W8A8 INT8 真量化 WikiText-2 PPL 评测。

## 文件清单

```
├── setup.sh                     一键脚本 (安装/量化/评测)
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

### 2. 一键执行

```bash
docker exec -it int8-eval bash
bash /data/int8-ppl-eval/setup.sh all /models/YourModel
```

或分步:
```bash
bash setup.sh install                      # 安装依赖
bash setup.sh quantize  /models/YourModel  # W8A8 量化
bash setup.sh eval_fp16 /models/YourModel  # FP16 PPL
bash setup.sh eval_w8a8 /models/YourModel  # W8A8 PPL
```

量化模型自动保存到 `/models/YourModel-W8A8`，结果保存到 `../results/`。

## 已知问题

| 问题 | 原因 | setup.sh 处理方式 |
|------|------|------------------|
| llmcompressor 降级 torch 到 2.8.0 | pip 依赖解析 | 自动备份+恢复 aarch64 wheel |
| `forward_oot() got unexpected keyword 'is_prefill'` | omni-npu 兼容性 | 自动 patch |
| `OMNI_NPU_PATCHES_DIR` 解析错误 | sys.argv 解析失败 | 自动从 config.json 检测 model_type |
| graph capture `stream is captured` 错误 | CANN 不兼容 | enforce_eager=True |
| engine 关闭 core dump | vllm 已知问题 | 结果不受影响，已处理 |

## 参考结果 (Qwen3-0.6B)

| 环境 | FP16 | W8A8 | 劣化 |
|------|------|------|------|
| omni-infer v1.0.0 | 60.0255 | 61.1840 | +1.93% |
