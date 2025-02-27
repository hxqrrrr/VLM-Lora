import torch
from transformers import AutoTokenizer
import os
from typing import Dict, Tuple, Optional, Union
from vlm_lora.models.tinyllama import TinyLLaMAForCausalLM
from vlm_lora.common.config import LoraConfig
from vlm_lora.adapters.lora.model import LoraModel
import torch.nn as nn

def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    return torch.device(device)

def get_model_path(local_base_path: str, default_path: str) -> str:
    local_path =  r"C:\Users\hxq11\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v0.4\snapshots\5378798738b1ec3ebf6f7c10c938540ff6dbb436"
    if os.path.exists(local_path):
        print(f"加载本地模型: {local_path}")
        return local_path
    print(f"本地模型不存在，将从Hugging Face下载: {default_path}")
    return default_path

def load_tokenizer(model_path: str) -> AutoTokenizer:
    print("\n加载分词器...")
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=os.path.exists(model_path),
            trust_remote_code=True
        )
    except Exception as e:
        raise RuntimeError(f"加载分词器失败: {str(e)}")

def check_lora_layers(model: LoraModel) -> None:
    print("\n=== 检查 LoRA 层状态 ===")
    if not model.lora_layers:
        print("警告: 未找到任何 LoRA 层，可能未正确替换目标模块！")
        return
    
    print(f"共找到 {len(model.lora_layers)} 个 LoRA 层")
    for name, layer in model.lora_layers.items():
        print(f"\nLoRA 层: {name}")
        print(f"  类型: {type(layer).__name__}")
        print(f"  lora_A.weight:")
        print(f"    requires_grad: {layer.lora_A.weight.requires_grad}")
        print(f"    形状: {layer.lora_A.weight.shape}")
        print(f"    设备: {layer.lora_A.weight.device}")
        print(f"  lora_B.weight:")
        print(f"    requires_grad: {layer.lora_B.weight.requires_grad}")
        print(f"    形状: {layer.lora_B.weight.shape}")
        print(f"    设备: {layer.lora_B.weight.device}")
        print(f"  base_layer:")
        print(f"    类型: {type(layer.base_layer).__name__}")
        print(f"    requires_grad: {layer.base_layer.weight.requires_grad}")
        print(f"    形状: {layer.base_layer.weight.shape}")
    print("\n=== 检查完成 ===")

def initialize_models(model_path: str, device: torch.device, lora_config: LoraConfig) -> Tuple[TinyLLaMAForCausalLM, LoraModel]:
    print("\n创建 TinyLlama 模型...")
    dtype = torch.float16
    base_model = TinyLLaMAForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype
    ).to(device)
    
    print("\n创建 LoRA 模型...")
    lora_model = LoraModel(
        base_model=base_model,
        config=lora_config,
        device=device
    )
    
    check_lora_layers(lora_model)
    return base_model, lora_model

def prepare_input(tokenizer: AutoTokenizer, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    print(f"\n输入文本: {text}")
    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

def test_forward(model: LoraModel, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    print("\n执行前向传播...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    print(f"输出形状: {outputs.shape}")
    print(f"输出是否包含 NaN: {torch.isnan(outputs).any()}")
    return outputs
def train_lora_step(
    model: LoraModel,
    tokenizer: AutoTokenizer,
    text: str,
    target: str,
    device: torch.device,
    max_length: int = 100
) -> float:
    print("\n执行 LoRA 微调一步...")
    model.train()
    
    combined_text = f"{text} {target}"
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs["input_ids"].to(device)  # 不设置 requires_grad=True
    attention_mask = inputs["attention_mask"].to(device)
    
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # 忽略最后一个 token
    
    # 设置优化器，只更新 LoRA 参数
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    # 前向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"原始 outputs requires_grad: {outputs.requires_grad}")
    
    shift_logits = outputs[:, :-1, :].contiguous()
    shift_labels = labels[:, :-1].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    print(f"loss requires_grad: {loss.requires_grad}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"微调损失: {loss.item():.4f}")
    return loss.item()

def generate_response(
    model: TinyLLaMAForCausalLM, 
    tokenizer: AutoTokenizer, 
    inputs: Dict[str, torch.Tensor], 
    max_length: int = 100
) -> str:
    print("\n生成回复...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    LOCAL_BASE_PATH = r"C:\Users\hxq11\.cache\huggingface\hub\models"
    DEFAULT_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
    TEXT = "你好，请解释一下什么是量子力学。"
    TARGET = "量子力学是物理学的一个重要分支，它描述了微观世界的行为规律。"
    
    try:
        device = get_device()
        model_path = get_model_path(LOCAL_BASE_PATH, DEFAULT_MODEL_PATH)
        tokenizer = load_tokenizer(model_path)
        
        lora_config = LoraConfig(
            target_modules_={"model.model.layers.0.self_attn.q_proj": True, "model.model.layers.0.self_attn.v_proj": True},
            lora_r_=8,
            lora_alpha_=16,
            lora_dropout_=0.1,
            use_dora_=False,
            adapter_name="tinyllama_lora"
        )
        
        base_model, lora_model = initialize_models(model_path, device, lora_config)
        inputs = prepare_input(tokenizer, TEXT, device)
        
        test_forward(lora_model, inputs)
        train_lora_step(lora_model, tokenizer, TEXT, TARGET, device)
        
        save_path = f"{lora_config.adapter_name}.pt"
        print(f"\n保存 LoRA 权重到 {save_path}...")
        
        
        response = generate_response(base_model, tokenizer, inputs)
        print(f"模型回复: {response}")
        
        print("\n测试完成!")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()