import torch
from transformers import AutoTokenizer
import os
import json
from vlm_lora.models.tinyllama import TinyLLaMAForCausalLM
from vlm_lora.common.config import LoraConfig, VLMModelConfig
from vlm_lora.adapters.lora.model import LoraModel


def main():
    # 加载配置文件
    print("加载配置文件...")
    with open("templates/tinyllama_lora.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    model_config = VLMModelConfig(**config["model_config"])
    lora_config = LoraConfig.from_config(config["lora_config"])
    
    # 设置设备
    device = model_config.device_
    print(f"使用设备: {device}")
    
    # 本地模型路径
    local_path = r"C:\Users\hxq11\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v0.4\snapshots\5378798738b1ec3ebf6f7c10c938540ff6dbb436"
    if not os.path.exists(local_path):
        local_path = model_config.name_or_path_
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
    
    # 创建基础模型
    print("\n创建 TinyLlama 模型")
    base_model = TinyLLaMAForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float16 if model_config.dtype_ == "float16" else torch.float32
    ).to(device)
    
    # 创建 LoRA 模型
    print("\n创建 LoRA 模型...")
    model = LoraModel(
        base_model=base_model,
        config=lora_config,
        device=device
    )
    
    # 准备训练数据
    text = "你好，请解释一下什么是量子力学。"
    target = "量子力学是物理学的一个重要分支，它描述了微观世界的行为规律。"
    print(f"\n训练数据:")
    print(f"输入: {text}")
    print(f"目标: {target}")
    
    # 编码输入和目标
    # 将输入和目标组合成一个序列
    combined_text = f"{text} {target}"
    print(f"\n组合后的文本: {combined_text}")
    
    # 编码
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=model_config.max_seq_len_)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 创建标签序列（shifted right）
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]  # 每个位置预测下一个token
    labels[:, -1] = -100  # 忽略最后一个位置的预测
    
    print(f"\n输入形状: {input_ids.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 训练一步
    print("\n执行一步训练...")
    model.train()
    
    # 确保 LoRA 参数可训练
    for layer in model.lora_layers.values():
        layer.lora_A.weight.requires_grad = True
        layer.lora_B.weight.requires_grad = True
        if layer.use_dora and layer.magnitude_vector is not None:
            layer.magnitude_vector.requires_grad = True
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    print(f"输出形状: {outputs.shape}")
    
    # 计算损失
    shift_logits = outputs[:, :-1, :].contiguous()
    shift_labels = labels[:, :-1].contiguous()
    
    print(f"处理后的logits形状: {shift_logits.shape}")
    print(f"处理后的labels形状: {shift_labels.shape}")
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    )
    
    # 反向传播
    loss.backward()
    print(f"训练损失: {loss.item():.4f}")
    
    # 保存 LoRA 权重
    save_path = f"{lora_config.adapter_name}.pt"
    print(f"\n保存 LoRA 权重到 {save_path}...")
    torch.save(model.get_lora_state_dict(), save_path)
    
    # 测试生成
    print("\n生成回复...")
    model.eval()
    with torch.no_grad():
        generated_ids = model.base_model.generate(
            inputs["input_ids"],
            max_length=model_config.max_seq_len_,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n模型回复: {response}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 