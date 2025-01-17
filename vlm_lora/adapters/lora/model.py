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
        
        print("\n配置的target_modules:")
        for name, value in self.config.target_modules_.items():
            print(f"  {name}: {value}")
            
        print("\n模型的模块结构:")
        for name, _ in self.base_model.named_modules():
            print(f"  {name}")
        
        # 冻结基础模型参数
        self.base_model.requires_grad_(False)
            
        # 替换目标模块为 LoRA 层
        self.lora_layers: Dict[str, LoraLayer] = {}
        print("\n开始替换模块...")
        replaced_count = self._replace_modules(self.base_model)
        print(f"\n成功替换 {replaced_count} 个层为 LoRA 层")
        print(f"替换的层: {list(self.lora_layers.keys())}")
        
        # 确保 LoRA 层参数可训练
        for layer in self.lora_layers.values():
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)
            if layer.use_dora and layer.magnitude_vector is not None:
                layer.magnitude_vector.requires_grad_(True)
        
    def _replace_modules(self, root_module: nn.Module, prefix: str = ""):
        """递归替换目标模块为 LoRA 层"""
        replaced_count = 0
        for name, module in root_module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            print(f"检查模块: {full_name} ({type(module).__name__})")
            
            # 如果是线性层，检查是否需要替换
            if isinstance(module, nn.Linear):
                # 检查模块名是否匹配任意目标模式
                should_replace = any(pattern in full_name for pattern in self.config.target_modules_)
                if should_replace:
                    print(f"替换层: {full_name}")
                    lora_layer = LoraLayer(
                        base_layer=module,
                        config=self.config,
                        device=self.device
                    )
                    
                    setattr(root_module, name, lora_layer)
                    self.lora_layers[full_name] = lora_layer
                    replaced_count += 1
            
            # 递归处理子模块
            sub_count = self._replace_modules(module, full_name)
            replaced_count += sub_count
                
        return replaced_count
        
    def forward(self, *args, **kwargs):
        """前向传播，使用替换后的LoRA层"""
        # 确保基础模型处于评估模式
        self.base_model.eval()
        
        # 确保所有LoRA层都处于正确的模式
        if self.training:
            for name, layer in self.lora_layers.items():
                print(f"设置 {name} 为训练模式")
                layer.train()
                # 确保LoRA参数可训练
                layer.lora_A.requires_grad_(True)
                layer.lora_B.requires_grad_(True)
                if layer.use_dora and layer.magnitude_vector is not None:
                    layer.magnitude_vector.requires_grad_(True)
        else:
            for layer in self.lora_layers.values():
                layer.eval()
        
        # 调用基础模型的前向传播
        outputs = self.base_model(*args, **kwargs)
        
        # 检查输出是否需要梯度
        print(f"\n输出张量是否需要梯度: {outputs.requires_grad}")
        print(f"输出张量的grad_fn: {outputs.grad_fn}")
        
        return outputs
        
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