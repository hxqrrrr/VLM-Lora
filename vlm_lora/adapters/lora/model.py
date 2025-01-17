import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from vlm_lora.adapters.lora.lora import LoraLayer
from vlm_lora.common.config import LoraConfig

class LoraModel(nn.Module):
    """LoRA 模型实现"""
    def __init__(
        self,
        base_model: nn.Module,
        config: LoraConfig,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 替换目标模块为 LoRA 层
        self.lora_layers: Dict[str, LoraLayer] = {}
        self._replace_modules(self.base_model)
        
    def _replace_modules(self, root_module: nn.Module, prefix: str = ""):
        """递归替换目标模块为 LoRA 层"""
        for name, module in root_module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 如果是目标模块且为线性层，替换为 LoRA 层
            if full_name in self.config.target_modules_ and isinstance(module, nn.Linear):
                if self.config.target_modules_[full_name]:
                    lora_layer = LoraLayer(
                        base_layer=module,
                        config=self.config,
                        device=self.device
                    )
                    
                    setattr(root_module, name, lora_layer)
                    self.lora_layers[full_name] = lora_layer
            else:
                # 递归处理子模块
                self._replace_modules(module, full_name)
                
    def forward(self, *args, **kwargs):
        """前向传播，直接调用基础模型的前向传播"""
        return self.base_model(*args, **kwargs)
        
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取 LoRA 层的状态字典"""
        state_dict = {}
        for name, layer in self.lora_layers.items():
            state_dict[f"{name}.lora_A.weight"] = layer.lora_A.weight
            state_dict[f"{name}.lora_B.weight"] = layer.lora_B.weight
            if layer.use_dora and layer.magnitude_vector is not None:
                state_dict[f"{name}.magnitude_vector"] = layer.magnitude_vector
        return state_dict
        
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """加载 LoRA 层的状态字典"""
        for name, layer in self.lora_layers.items():
            if f"{name}.lora_A.weight" in state_dict:
                layer.lora_A.weight.data.copy_(state_dict[f"{name}.lora_A.weight"])
            if f"{name}.lora_B.weight" in state_dict:
                layer.lora_B.weight.data.copy_(state_dict[f"{name}.lora_B.weight"])
            if layer.use_dora and f"{name}.magnitude_vector" in state_dict:
                layer.magnitude_vector.data.copy_(state_dict[f"{name}.magnitude_vector"]) 