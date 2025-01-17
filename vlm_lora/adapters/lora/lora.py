import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlm_lora.common.config import LoraConfig


class LoraLayer(nn.Module):
    """LoRA 层实现"""
    def __init__(
        self,
        base_layer: nn.Module,
        config: LoraConfig,
        device: str = None
    ):
        super().__init__()
        
        # 保存基础层
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)  # 冻结基础层参数
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        # 获取输入输出维度
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            raise ValueError(f"不支持的层类型: {type(base_layer)}")
            
        # LoRA 超参数
        self.r = config.lora_r_
        self.alpha = config.lora_alpha_
        if config.use_rslora_:
            self.scaling = self.alpha / math.sqrt(self.r)
        else:
            self.scaling = self.alpha / self.r
        
        # 创建 LoRA 层
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False)
        self.dropout = nn.Dropout(p=config.lora_dropout_)
        
        # DoRA 相关
        self.use_dora = config.use_dora_
        self.magnitude_vector = None
        
        # 初始化
        self.reset_parameters(config.lora_init_)
        
    def _get_weight_norm(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """计算权重矩阵的 L2 范数"""
        weight = self.base_layer.weight.to(dtype)
        lora_weight = self.lora_B.weight @ self.lora_A.weight
        weight = weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm
        
    def reset_parameters(self, init_type: str = "original"):
        """重置 LoRA 参数"""
        if init_type == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.r)
        else:  # original
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        if self.use_dora:
            self.magnitude_vector = nn.Parameter(
                self._get_weight_norm(), requires_grad=True
            )
            
    def apply_dora(self, residual: torch.Tensor, result_lora: torch.Tensor):
        """应用 DoRA"""
        weight_norm = self._get_weight_norm().detach()
        mag_norm_scale = (self.magnitude_vector / weight_norm).view(1, -1)
        return mag_norm_scale * residual + mag_norm_scale * result_lora
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础层前向传播
        base_output = self.base_layer(x)
        
        # LoRA 前向传播
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        # 合并输出
        if self.use_dora:
            return self.apply_dora(base_output, lora_output)
        else:
            return base_output + lora_output
