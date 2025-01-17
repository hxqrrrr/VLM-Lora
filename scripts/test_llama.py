import torch
from transformers import LlamaForCausalLM as HFLlama
from transformers import LlamaTokenizer

from vlm_lora.models.llama import LlamaForCausalLM


def main():
    # 加载原始模型和分词器
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
    print(f"加载模型: {model_path}")
    hf_model = HFLlama.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    # 创建我们的模型
    print("创建 VLM-LoRA 模型")
    model = LlamaForCausalLM.from_pretrained(hf_model)
    
    # 准备输入
    text = "Hello, how are you?"
    print(f"\n输入文本: {text}")
    inputs = tokenizer(text, return_tensors="pt")
    
    # 测试前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    
    # 验证输出
    logits = outputs.logits
    print(f"输出logits形状: {logits.shape}")
    print(f"输出是否包含 NaN: {torch.isnan(logits).any()}")
    print("测试完成!")


if __name__ == "__main__":
    main() 