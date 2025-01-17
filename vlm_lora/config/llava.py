from dataclasses import dataclass
from vlm_lora.common.config import VLMModelConfig
from typing import Dict, Optional




@dataclass
class LLaVAModelConfig(VLMModelConfig):
    # LLaVA需要额外的视觉相关配置
    vision_tower_: str = None          # 视觉编码器路径
    vision_tower_dim_: int = None      # 视觉特征维度
    mm_vision_select_layer_: int = None # 视觉特征选择层
    mm_projector_type_: str = None     # 多模态投影器类型
    image_token_len_: int = None       # 图像token长度
    image_aspect_ratio_: str = None    # 图像长宽比处理策略