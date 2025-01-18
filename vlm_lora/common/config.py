import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeAlias, Union
import json
import torch

Tokens: TypeAlias = List[int]
Labels: TypeAlias = List[int]
Masks: TypeAlias = List[bool]

@dataclass
class Prompt:
    instruction: str = None
    input: str = None
    label: str = None


@dataclass
class InputData:
    inputs: List[Union[Prompt, List[str], str]] = None
    tokens: Optional[Tokens] = None
    labels: Optional[Labels] = None

    
@dataclass
class VLMModelOutput:
    adapter_name: str = None
    logits: torch.Tensor = None
    router_logits: torch.Tensor = None
    loss: torch.Tensor = None
    aux_loss: torch.Tensor = None
    # for internal use
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    loss_fn_: Callable = None


@dataclass
class VLMModelConfig:
    name_or_path_: str = None
    device_: str = None
    dim_: int = None
    head_dim_: int = None
    intermediate_: int = None
    n_heads_: int = None
    n_kv_heads_: int = None
    n_layers_: int = None
    hidden_act_: str = None
    hidden_dropout_: float = None
    vocab_size_: int = None
    pad_token_id_: int = None
    rope_theta_: float = None
    partial_rotary_factor_: float = None
    max_seq_len_: int = None
    # eager or flash_attn
    attn_implementation_: str = "eager"
    # data type
    dtype_: torch.dtype = None

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "VLMModelConfig":
        raise NotImplementedError("Subclasses must implement this method.")


@dataclass
class BatchConfig:
    adapter_name_: str = ""
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1


@dataclass
class VLMModelInput:
    batch_configs_: List[BatchConfig] = None
    batch_tokens_: List[Tokens] = None
    batch_labels_: List[Labels] = None
    batch_masks_: List[Masks] = None

    output_router_logits_: bool = True

    gradient_checkpoint_: str = "none"
    efficient_operator_: bool = False
    inference_mode_: bool = False


@dataclass
class AdapterConfig:
    adapter_name: str = ""
    task_name: str = "casual"

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            adapter_name=config.get("name", None),
            task_name=config.get("task_name", None),
        )


lora_target_modules = {
    # VLM names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "o_proj": False,
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
    # Phi names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "dense": False,
    "fc1": False,
    "fc2": False,
    # Phi3 names
    "qkv_proj": False,
    "o_proj": False,
    "gate_up_proj": False,
    "down_proj": False,
    # GLM names
    "qkv_proj": False,
    "dense": False,
    "dense_h_to_4h": False,
    "dense_4h_to_h": False,
}

@dataclass
class LoraConfig(AdapterConfig):
    # Weight-Decomposed Low-Rank Adaptation
    lora_r_: int = 8
    lora_alpha_: int = 16
    lora_dropout_: float = 0.0
    use_dora_: bool = False
    use_rslora_: bool = False
    lora_init_: str = "original"
    target_modules_: Dict[str, bool] = field(default_factory=dict)

    def export(self) -> Dict[str, any]:
        config = {}
        if self.use_dora_:
            config["use_dora"] = True
        if self.use_rslora_:
            config["use_rslora"] = True
        config["bias"] = "none"
        config["peft_type"] = "LORA"
        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        tgt_list = []
        for target, value in self.target_modules_.items():
            if value:
                tgt_list.append(target)
        config["target_modules"] = tgt_list
        return config
    
    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraConfig":
        lora_config = LoraConfig()
        
        # 基础配置
        lora_config.adapter_name_ = config.get("adapter_name_", "")
        lora_config.task_name_ = config.get("task_name_", "casual")
        
        # LoRA参数
        lora_config.lora_r_ = config.get("lora_r_", 8)
        lora_config.lora_alpha_ = config.get("lora_alpha_", 16)
        lora_config.lora_dropout_ = config.get("lora_dropout_", 0.0)
        
        # 特殊功能开关
        lora_config.use_dora_ = config.get("use_dora_", False)
        lora_config.use_rslora_ = config.get("use_rslora_", False)
        lora_config.lora_init_ = config.get("lora_init_", "original")
        
        # 处理目标模块
        target_modules = config.get("target_modules_", [])
        if isinstance(target_modules, list):
            lora_config.target_modules_ = {module: True for module in target_modules}
        else:
            lora_config.target_modules_ = target_modules
        
        return lora_config
    
    @staticmethod
    def from_json(json_path: str) -> "LoraConfig":
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return LoraConfig.from_config(config_dict.get("lora_config", {}))
