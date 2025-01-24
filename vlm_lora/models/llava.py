import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from transformers import LlamaForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor

from vlm_lora.common import (
    VLMModelConfig,
    VLMCache,
    VLMDecoder,
    VLMForCausalLM,
)

@dataclass
class LLaVAConfig(VLMModelConfig):
    """LLaVA 模型配置类"""
    # 基础配置
    name_or_path_: str = "liuhaotian/llava-v1.5-7b"  # 模型路径
    vision_tower_: str = "openai/clip-vit-large-patch14"
    mm_vision_select_layer_: int = -2
    mm_vision_select_feature_: str = "patch"
    mm_hidden_size_: int = 1024
    
    # 模型配置
    hidden_size_: int = 4096  # LLaVA-1.5-7B 的隐藏层大小
    num_attention_heads_: int = 32
    num_hidden_layers_: int = 32
    max_seq_len_: int = 4096
    
    # 训练配置
    tune_mm_mlp_adapter_: bool = False
    mm_use_im_start_end_: bool = True
    
    # 设备和数据类型配置
    dtype_: torch.dtype = torch.float16
    device_map_: Optional[str] = "auto"
    
    def from_json(cls, json_data: Dict[str, Any]) -> "LLaVAConfig":
        """从JSON数据创建LLaVA配置对象
        
        Args:
            json_data: 包含配置参数的字典
            
        Returns:
            LLaVAConfig: 新的配置对象
        """
        # 创建配置参数字典
        config_kwargs = {}
        
        # 遍历类的所有属性
        for field_name, field_value in cls.__dataclass_fields__.items():
            # 如果JSON中存在该字段，则使用JSON中的值
            if field_name in json_data:
                value = json_data[field_name]
                
                # 处理特殊类型
                if field_name.endswith('_dtype'):
                    if isinstance(value, str):
                        value = getattr(torch, value)
                
                config_kwargs[field_name] = value
                
        # 创建并返回新的配置对象
        return cls(**config_kwargs)

class LLaVAVisionTower(nn.Module):
    """视觉编码器塔"""
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config
        
        # 加载CLIP视觉模型
        self.vision_model = CLIPVisionModel.from_pretrained(
            config.vision_tower_,
            torch_dtype=config.dtype_,
            device_map=config.device_map_
        )
        self.vision_model.requires_grad_(False)
        
        # 加载图像处理器
        self.image_processor = CLIPImageProcessor.from_pretrained(config.vision_tower_)
        
    def forward(self, images: torch.FloatTensor) -> Dict[str, torch.FloatTensor]:
        outputs = self.vision_model(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states[self.config.mm_vision_select_layer_]
        
        if self.config.mm_vision_select_feature_ == "patch":
            image_features = hidden_states[:, 1:]  # 移除[CLS]
        else:
            image_features = hidden_states
            
        return {"image_features": image_features}
    
    def preprocess(self, images: Union[torch.FloatTensor, List[str]]) -> torch.FloatTensor:
        if isinstance(images, torch.Tensor):
            return images
            
        image_tensors = self.image_processor(
            images,
            return_tensors="pt",
            do_resize=True,
            do_center_crop=True,
            size={"height": 224, "width": 224}
        ).pixel_values
        
        return image_tensors.to(device=self.vision_model.device, dtype=self.vision_model.dtype)

class LLaVAProjector(nn.Module):
    """多模态投影层"""
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.mm_hidden_size_, config.hidden_size_, bias=True)
        self.layer_norm = nn.LayerNorm(config.hidden_size_)
        self.linear_2 = nn.Linear(config.hidden_size_, config.hidden_size_, bias=True)
        
        if not config.tune_mm_mlp_adapter_:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.layer_norm(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear_2(x)
        return x

class LLaVADecoder(VLMDecoder):
    """LLaVA专用解码器"""
    def __init__(self, model: "LLaVA", max_length: int = 4096):
        super().__init__()
        self.model = model
        self.max_length = max_length
        
    def generate(
        self,
        model: "LLaVA",
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        cache: Optional[VLMCache] = None,
        **kwargs
    ) -> torch.Tensor:
        """生成文本"""
        # 使用transformers的生成方法
        if inputs_embeds is not None:
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=model.tokenizer.pad_token_id,
                bos_token_id=model.tokenizer.bos_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                **kwargs
            )
        else:
            outputs = model.language_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=model.tokenizer.pad_token_id,
                bos_token_id=model.tokenizer.bos_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                **kwargs
            )
        return outputs

class LLaVA(VLMForCausalLM):
    """LLaVA 模型实现"""
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config
        
        # 加载语言模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path_)
        self.language_model = LlamaForCausalLM.from_pretrained(
            config.name_or_path_,
            torch_dtype=config.dtype_,
            device_map=config.device_map_
        )
        
        # 加载视觉模型
        self.vision_tower = LLaVAVisionTower(config)
        
        # 初始化投影层
        self.mm_projector = LLaVAProjector(config)
        
        # 添加特殊token
        special_tokens = {
            "additional_special_tokens": [
                "<image>",
                "</image>",
                "<image_newline>"
            ]
        }
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if num_new_tokens > 0:
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            
        # 初始化解码器
        self.decoder = LLaVADecoder(self, config.max_seq_len_)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[Union[torch.FloatTensor, List[str]]] = None,
        labels: Optional[torch.Tensor] = None,
        cache: Optional[VLMCache] = None,
    ) -> Dict[str, torch.Tensor]:
        if images is not None:
            # 处理图像
            image_tensors = self.vision_tower.preprocess(images)
            image_features = self.vision_tower(image_tensors)["image_features"]
            projected_features = self.mm_projector(image_features)
            
            # 获取图像token位置
            image_start_tokens = (input_ids == self.tokenizer.convert_tokens_to_ids("<image>")).nonzero()
            image_end_tokens = (input_ids == self.tokenizer.convert_tokens_to_ids("</image>")).nonzero()
            
            # 获取词嵌入
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # 插入图像特征
            for idx, (start_idx, end_idx) in enumerate(zip(image_start_tokens, image_end_tokens)):
                inputs_embeds[idx, start_idx+1:end_idx] = projected_features[idx]
                
            # 使用嵌入进行前向传播
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=cache.past_key_values if cache else None,
                use_cache=True,
                return_dict=True
            )
        else:
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=cache.past_key_values if cache else None,
                use_cache=True,
                return_dict=True
            )
            
        # 更新缓存
        if cache is not None:
            cache.update(outputs.past_key_values)
            
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[Union[torch.FloatTensor, List[str]]] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        # 初始化缓存
        cache = VLMCache()
        
        if images is not None:
            image_tensors = self.vision_tower.preprocess(images)
            image_features = self.vision_tower(image_tensors)["image_features"]
            projected_features = self.mm_projector(image_features)
            
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            image_start_tokens = (input_ids == self.tokenizer.convert_tokens_to_ids("<image>")).nonzero()
            image_end_tokens = (input_ids == self.tokenizer.convert_tokens_to_ids("</image>")).nonzero()
            
            for idx, (start_idx, end_idx) in enumerate(zip(image_start_tokens, image_end_tokens)):
                inputs_embeds[idx, start_idx+1:end_idx] = projected_features[idx]
                
            # 使用解码器生成
            return self.decoder.generate(
                model=self,
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                cache=cache,
                **kwargs
            )
        else:
            return self.decoder.generate(
                model=self,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                cache=cache,
                **kwargs
            )

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "LLaVA":
        config = LLaVAConfig(name_or_path_=model_path, **kwargs)
        return cls(config)
        
    def get_tokenizer(self) -> AutoTokenizer:
        """获取分词器"""
        return self.tokenizer
        
    def get_decoder(self) -> VLMDecoder:
        """获取解码器"""
        return self.decoder 
    
