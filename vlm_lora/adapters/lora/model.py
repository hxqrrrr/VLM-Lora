import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from vlm_lora.adapters.lora.lora import LoraLayer
from vlm_lora.common.config import LoraConfig

class LoraModel(nn.Module):
    def __init__(self, base_model: nn.Module, config: LoraConfig, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        # 选择性冻结基础模型参数，但保留嵌入层的梯度
        for name, param in self.base_model.named_parameters():
            if "embed_tokens" not in name:  # 只冻结非嵌入层参数
                param.requires_grad_(False)
        
        
        
        self.lora_layers: Dict[str, LoraLayer] = {}
        replaced_count = self._replace_modules(self.base_model)
        print(f"成功替换 {replaced_count} 个层为 LoRA 层")
        
        # 启用 LoRA 层参数的梯度
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
                lora_layer = LoraLayer(module, self.config, self.device)
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
        print(f"前向传播输出 requires_grad: {outputs.requires_grad}")
        return outputs
