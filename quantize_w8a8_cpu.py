"""
W8A8 INT8 量化脚本 (CPU, 无需 GPU/NPU)
使用 llmcompressor 0.8.1 RTN (Round-To-Nearest)，data-free。

用法:
  python3 quantize_w8a8_cpu.py --model /path/to/model --output /path/to/output

量化配置:
  权重: INT8, symmetric, per-channel static
  激活: INT8, symmetric, per-token dynamic
  排除: lm_head
"""
import argparse
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description="W8A8 INT8 CPU quantization")
    parser.add_argument("--model", required=True, help="FP16/BF16 model path")
    parser.add_argument("--output", required=True, help="Output directory for quantized model")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"Loading model from {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["lm_head"],
    )
    print("Applying W8A8 quantization (RTN, data-free) ...")
    oneshot(model=model, recipe=recipe)

    print(f"Saving to {args.output} ...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
