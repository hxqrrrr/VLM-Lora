import json
import torch
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from .config import VLMModelConfig, LoraConfig
from ..adapters import lora_config_factory


def load_json_config(config_path: str) -> Tuple[VLMModelConfig, LoraConfig]:
    """
    从JSON文件加载配置
    
    Args:
        config_path: JSON配置文件的路径
        
    Returns:
        Tuple[VLMModelConfig, LoraConfig]: 返回模型配置和LoRA配置
    """
    # 读取JSON文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # 处理模型配置
    model_config_dict = config_dict.get("model_config", {})
    if "dtype_" in model_config_dict:
        # 将字符串类型转换为torch.dtype
        dtype_str = model_config_dict["dtype_"]
        if dtype_str == "float32":
            model_config_dict["dtype_"] = torch.float32
        elif dtype_str == "float16":
            model_config_dict["dtype_"] = torch.float16
        elif dtype_str == "bfloat16":
            model_config_dict["dtype_"] = torch.bfloat16
    
    # 创建VLMModelConfig实例
    model_config = VLMModelConfig(**model_config_dict)
    
    # 处理LoRA配置
    lora_config_dict = config_dict.get("lora_config", {})
    
    # 使用lora_config_factory创建LoRA配置
    lora_config = lora_config_factory(lora_config_dict)
    
    return model_config, lora_config



def print_config(model_config: VLMModelConfig, lora_config: LoraConfig) -> None:
    """
    打印配置信息
    
    Args:
        model_config: 模型配置
        lora_config: LoRA配置
    """
    print("\n" + "="*50)
    print("模型配置:")
    print("="*50)
    print(f"模型名称: {model_config.name_or_path_}")
    print(f"设备: {model_config.device_}")
    print(f"数据类型: {model_config.dtype_}")
    print(f"隐藏层维度: {model_config.dim_}")
    print(f"注意力头数量: {model_config.n_heads_}")
    print(f"Transformer层数: {model_config.n_layers_}")
    print(f"最大序列长度: {model_config.max_seq_len_}")
    
    print("\n" + "="*50)
    print("LoRA配置:")
    print("="*50)
    print(f"适配器名称: {lora_config.adapter_name}")
    print(f"任务类型: {lora_config.task_name}")
    print(f"LoRA秩(r): {lora_config.lora_r_}")
    print(f"缩放因子(alpha): {lora_config.lora_alpha_}")
    print(f"Dropout率: {lora_config.lora_dropout_}")
    print("\n目标模块:")
    for module, enabled in lora_config.target_modules_.items():
        if enabled:
            print(f"  - {module}")
    print("="*50)
