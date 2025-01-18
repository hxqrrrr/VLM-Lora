from typing import Dict, Type
from torch.utils.data import Dataset
from .dataset_config import DatasetConfig
from .load_dataset import DTDDataset

class DatasetFactory:
    """数据集工厂类"""
    
    _dataset_registry: Dict[str, Type[Dataset]] = {
        "DTD": DTDDataset
    }
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: Type[Dataset]):
        """注册新的数据集类"""
        cls._dataset_registry[name] = dataset_class
        
    @classmethod
    def create_dataset(cls, config: DatasetConfig, split: str = "train") -> Dataset:
        """
        创建数据集实例
        Args:
            config: 数据集配置
            split: 数据集分割（train/val/test）
        Returns:
            Dataset实例
        """
        if config.dataset_name_ not in cls._dataset_registry:
            raise ValueError(f"未知的数据集类型: {config.dataset_name_}")
            
        dataset_class = cls._dataset_registry[config.dataset_name_]
        return dataset_class(
            root_dir=config.dataset_path_,
            split=split,
            image_size=config.image_size_,
            image_mean=config.image_mean_,
            image_std=config.image_std_,
            max_length=config.max_length_
        ) 