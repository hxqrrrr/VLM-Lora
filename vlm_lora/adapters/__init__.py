from typing import Dict
from vlm_lora.common import LoraConfig

def lora_config_factory(config: Dict[str, any]) -> LoraConfig:
    """
    创建 LoRA 配置实例
    
    Args:
        config: 配置字典
        
    Returns:
        LoraConfig: LoRA 配置实例
    """
    return LoraConfig.from_config(config).check()
