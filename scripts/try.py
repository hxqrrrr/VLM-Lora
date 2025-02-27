import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
from typing import Dict, Tuple, Optional, Union

# 假设的 LoraConfig 和 TinyLLaMAForCausalLM（需要从 vlm_lora 导入）
from vlm_lora.models.tinyllama import TinyLLaMAForCausalLM
from vlm_lora.common.config import LoraConfig

class LoraLayer(nn.Module):
    def __init__(self, base_layer: nn.Linear, config: LoraConfig, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Linear(in_features, config.lora_r_, bias=False, device=self.device)
        self.lora_B = nn.Linear(config.lora_r_, out_features, bias=False, device=self.device)
        self.scaling = config.lora_alpha_ / config.lora_r_
        self.dropout = nn.Dropout(p=config.lora_dropout_)
        
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.normal_(self.lora_B.weight, std=0.02)
        
        self.use_dora = config.use_dora_
        if self.use_dora:
            self.magnitude_vector = nn.Parameter(torch.ones(out_features, device=self.device))
        else:
            self.magnitude_vector = None
            
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        print(f"输入 x requires_grad: {x.requires_grad}")
        
        base_output = self.base_layer(x)
        print(f"base_output requires_grad: {base_output.requires_grad}")
        
        lora_input = self.dropout(x)
        lora_A_out = self.lora_A(lora_input)
        print(f"lora_A_out requires_grad: {lora_A_out.requires_grad}")
        
        lora_output = self.lora_B(lora_A_out) * self.scaling
        print(f"lora_output requires_grad: {lora_output.requires_grad}")
        
        if self.use_dora and self.magnitude_vector is not None:
            lora_output = lora_output * self.magnitude_vector
            print(f"lora_output (DoRA) requires_grad: {lora_output.requires_grad}")
        
        final_output = base_output + lora_output
        print(f"final_output requires_grad: {final_output.requires_grad}")
        return final_output

class LoraModel(nn.Module):
    def __init__(self, base_model: nn.Module, config: LoraConfig, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        self.base_model.requires_grad_(False)
        self.lora_layers: Dict[str, LoraLayer] = {}
        replaced_count = self._replace_modules(self.base_model)
        print(f"成功替换 {replaced_count} 个层为 LoRA 层")
        
        for name, layer in self.lora_layers.items():
            layer.lora_A.weight.requires_grad_(True)
            layer.lora_B.weight.requires_grad_(True)
            if layer.use_dora and layer.magnitude_vector is not None:
                layer.magnitude_vector.requires_grad_(True)
    
    def _replace_modules(self, root_module: nn.Module, prefix: str = ""):
        replaced_count = 0
        for name, module in root_module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, nn.Linear) and any(pattern in full_name for pattern in self.config.target_modules_):
                lora_layer = LoraLayer(
                    base_layer=module,
                    config=self.config,
                    device=self.device
                )
                setattr(root_module, name, lora_layer)
                self.lora_layers[full_name] = lora_layer
                replaced_count += 1
            replaced_count += self._replace_modules(module, full_name)
        return replaced_count
    
    def forward(self, *args, **kwargs):
        if self.training:
            self.base_model.train()
            for name, layer in self.lora_layers.items():
                print(f"设置 {name} 为训练模式")
                layer.train()
        else:
            self.base_model.eval()
            for layer in self.lora_layers.values():
                layer.eval()
        
        outputs = self.base_model(*args, **kwargs)
        print(f"base_model 输出 requires_grad: {outputs.requires_grad}")
        return outputs
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for name, layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A.weight"] = layer.lora_A.weight
            state_dict[f"{name}.lora_B.weight"] = layer.lora_B.weight
            if layer.use_dora and layer.magnitude_vector is not None:
                state_dict[f"{name}.magnitude_vector"] = layer.magnitude_vector
        return state_dict

def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    return torch.device(device)

def get_model_path(local_base_path: str, default_path: str) -> str:
    local_path = r"C:\Users\hxq11\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v0.4\snapshots\5378798738b1ec3ebf6f7c10c938540ff6dbb436"
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
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"原始 outputs 类型: {type(outputs)}")
    print(f"原始 outputs requires_grad: {outputs.requires_grad}")
    
    shift_logits = outputs[:, :-1, :].contiguous()
    shift_labels = labels[:, :-1].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    print(f"loss requires_grad: {loss.requires_grad}")
    loss.backward()
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
        torch.save(lora_model.get_lora_state_dict(), save_path)
        
        response = generate_response(base_model, tokenizer, inputs)
        print(f"模型回复: {response}")
        
        print("\n测试完成!")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()