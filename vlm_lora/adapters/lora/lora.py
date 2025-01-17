import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlm_lora.common.config import LoraConfig
from typing import Optional, Union


class LoraLayer(nn.Module):
    """LoRA 层实现"""
    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoraConfig,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        # 获取基础层的输入和输出维度
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # 创建 LoRA 参数
        self.lora_A = nn.Linear(in_features, config.lora_r_, bias=False, device=self.device)
        self.lora_B = nn.Linear(config.lora_r_, out_features, bias=False, device=self.device)
        self.scaling = config.lora_alpha_ / config.lora_r_
        self.dropout = nn.Dropout(p=config.lora_dropout_)
        
        # 初始化 LoRA 参数
        if config.lora_init_ == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=0.02)
            nn.init.normal_(self.lora_B.weight, std=0.02)
        else:  # original
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        
        # DoRA 相关参数
        self.use_dora = config.use_dora_
        if self.use_dora:
            self.magnitude_vector = nn.Parameter(torch.ones(out_features, device=self.device))
        else:
            self.magnitude_vector = None
            
        # 确保所有参数都在正确的设备上
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 基础层的输出
        base_output = self.base_layer(x)
        
        # LoRA 路径
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        # 如果使用 DoRA，应用 magnitude vector
        if self.use_dora and self.magnitude_vector is not None:
            lora_output = lora_output * self.magnitude_vector
            
        return base_output + lora_output
