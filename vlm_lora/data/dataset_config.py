from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json

@dataclass
class DatasetConfig:
    """数据集配置类"""
    dataset_name_: str  # 数据集名称
    dataset_path_: str  # 数据集路径
    dataset_type_: str  # 数据集类型 (image_text_pair, image_only, text_only)
    image_size_: Optional[int] = 224  # 图像大小
    image_mean_: Optional[List[float]] = None  # 图像均值
    image_std_: Optional[List[float]] = None   # 图像标准差
    max_length_: Optional[int] = 77  # 文本最大长度
    
    @classmethod
    def from_json(cls, json_file: str) -> "DatasetConfig":
        """从JSON文件加载配置"""
        with open(json_file, "r") as f:
            config = json.load(f)
        return cls(**config.get("dataset_config", {}))

    def __post_init__(self):
        """初始化后的处理"""
        if self.image_mean_ is None:
            self.image_mean_ = [0.48145466, 0.4578275, 0.40821073]  # CLIP默认值
        if self.image_std_ is None:
            self.image_std_ = [0.26862954, 0.26130258, 0.27577711]  # CLIP默认值 