import torch
from transformers import AutoTokenizer
import os
from vlm_lora.models.tinyllama import TinyLLaMAForCausalLM


def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 本地模型路径
    local_path = r"C:\Users\hxq11\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v0.4\snapshots\5378798738b1ec3ebf6f7c10c938540ff6dbb436"
    if not os.path.exists(local_path):
        local_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
        print(f"模型不存在于本地，正在从huggingface下载: {local_path}")
    else:
        print(f"加载模型: {local_path}")
    # 加载分词器
    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # 创建我们的模型
    print("\n创建 TinyLlama 模型")
    model = TinyLLaMAForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float16  # 使用半精度以节省显存
    ).to(device)
    
    # 准备输入
    text = "你好，请解释一下什么是量子力学。"
    print(f"\n输入文本: {text}")
    inputs = tokenizer(text, return_tensors="pt")
    # 将输入移到正确的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 测试前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    
    # 验证输出
    logits = outputs
    print(f"\n输出形状: {logits.shape}")
    print(f"输出是否包含 NaN: {torch.isnan(logits).any()}")
    
    # 生成回复
    print("\n生成回复...")
    generated_ids = model.model.generate(
        inputs["input_ids"],
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n模型回复: {response}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 