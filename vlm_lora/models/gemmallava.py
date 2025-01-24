from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import (
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers import (
    AutoConfig,
    GemmaConfig,
    CLIPVisionModel,
    GemmaForCausalLM,
)

from vlm_lora.common import (
    ROPE_INIT_FUNCTIONS,
    FeedForward,
    Linear,
    VLMAttention,
    VLMCache,
    VLMDecoder,
    VLMFeedForward,
    VLMForCausalLM,
    VLMModelConfig,
    VLMModelInput,
    eager_attention_forward,
)
from vlm_lora.executors import executor
from vlm_lora.utils import copy_parameters

import math

@dataclass
class GemmaLavaConfig(VLMModelConfig):
    # 基础模型配置
    hidden_size_: int = 2048  # Gemma-2b 的隐藏层大小
    num_attention_heads_: int = 32
    num_hidden_layers_: int = 18
    max_seq_len_: int = 8192
    vocab_size_: int = 256000
    
    # 模型特定配置
    head_dim_: int = 64  # hidden_size_ // num_attention_heads_
    intermediate_size_: int = 5632  # MLP 中间层大小
    rope_theta_: float = 10000  # RoPE 位置编码参数
    rms_norm_eps_: float = 1e-6  # LayerNorm 参数
    
    # 视觉相关配置
    vision_tower_: str = "openai/clip-vit-large-patch14"  # 视觉模型路径
    vision_hidden_size_: int = 1024  # CLIP 隐藏层大小
    image_token_len_: int = 576  # 图像序列长度
    
    def __post_init__(self):
        """确保配置参数的一致性"""
        # 确保 head_dim 正确
        if self.head_dim_ * self.num_attention_heads_ != self.hidden_size_:
            self.head_dim_ = self.hidden_size_ // self.num_attention_heads_
            
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "GemmaLavaConfig":
        """从预训练模型创建配置"""
        return cls(
            name_or_path_=model_path,
            device_=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            dtype_=kwargs.get("dtype", torch.float32)
        )

@dataclass
class GemmaLavaModelInput(VLMModelInput):
    """GemmaLava 模型的输入参数类"""
    # 基本参数
    batch_size: int = None
    seq_length: int = None
    dtype: torch.dtype = None
    device: str = None
    
    # 运行时参数
    inference_mode: bool = True
    
    @classmethod
    def from_config(
        cls,
        batch_size: int,
        seq_length: int,
        config: GemmaLavaConfig,
        inference_mode: bool = True
    ) -> "GemmaLavaModelInput":
        """从配置创建输入参数对象"""
        return cls(
            batch_size=batch_size,
            seq_length=seq_length,
            dtype=config.dtype_,
            device=config.device_,
            inference_mode=inference_mode,
            gradient_checkpoint_="none",
            efficient_operator_=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[VLMCache]] = None,
        image_features: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(
                seq_length,
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0)
            
        # 1. 获取输入嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. 如果有图像特征，拼接到文本特征前面
        if image_features is not None:
            hidden_states = torch.cat([image_features, hidden_states], dim=1)
            # 更新attention mask
            if attention_mask is not None:
                image_mask = torch.ones(
                    (batch_size, image_features.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # 3. 获取旋转位置编码
        rotary_emb = self.rotary_embed(hidden_states, position_ids)
        
        # 4. 创建输入参数对象
        input_args = GemmaLavaModelInput.from_config(
            batch_size=batch_size,
            seq_length=seq_length,
            config=self.config_,
            inference_mode=True
        )
        
        # 5. 通过解码器层
        for idx, decoder_layer in enumerate(self.layers_):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                input_args,
                rotary_emb,
                attention_mask,
                None,
                layer_past,
            )
            
        # 6. 最后的层归一化
        hidden_states = self.norm(hidden_states)
        
        # 7. 语言模型头部
        logits = self.lm_head_(hidden_states)
        
        return logits

class GemmaLavaEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)
        return data

class GemmaLavaRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)
        return (self.weight_ * data).to(input_dtype)

class GemmaLavaAttention(nn.Module):
    def __init__(self, config: GemmaLavaConfig):
        super().__init__()
        self.hidden_size_ = config.hidden_size_
        self.num_heads_ = config.num_attention_heads_
        self.head_dim_ = config.head_dim_
        
        # 投影层
        self.q_proj_ = nn.Linear(self.hidden_size_, self.num_heads_ * self.head_dim_, bias=False)
        self.k_proj_ = nn.Linear(self.hidden_size_, self.num_heads_ * self.head_dim_, bias=False)
        self.v_proj_ = nn.Linear(self.hidden_size_, self.num_heads_ * self.head_dim_, bias=False)
        self.o_proj_ = nn.Linear(self.num_heads_ * self.head_dim_, self.hidden_size_, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 投影查询、键、值
        query = self.q_proj_(hidden_states)
        key = self.k_proj_(hidden_states)
        value = self.v_proj_(hidden_states)
        
        # 重塑形状以适应多头注意力
        query = query.view(batch_size, seq_length, self.num_heads_, self.head_dim_).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads_, self.head_dim_).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads_, self.head_dim_).transpose(1, 2)
        
        # 应用旋转位置编码
        if rotary_emb is not None:
            cos, sin = rotary_emb
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            
        # 如果有过去的键值对,拼接它们
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
            
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim_)
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value)
        
        # 重塑并投影输出
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_length, -1)
        hidden_states = self.o_proj_(hidden_states)
        
        return hidden_states

class GemmaLavaMLP(nn.Module):
    def __init__(self, config: GemmaLavaConfig):
        super().__init__()
        self.gate_proj_ = nn.Linear(config.hidden_size_, config.intermediate_size_, bias=False)
        self.up_proj_ = nn.Linear(config.hidden_size_, config.intermediate_size_, bias=False)
        self.down_proj_ = nn.Linear(config.intermediate_size_, config.hidden_size_, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj_(x))
        up = self.up_proj_(x)
        return self.down_proj_(gate * up)

class GemmaLavaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaLavaConfig):
        super().__init__()
        self.self_attn_ = GemmaLavaAttention(config)
        self.mlp_ = GemmaLavaMLP(config)
        self.input_layernorm_ = GemmaLavaRMSNorm(
            torch.ones(config.hidden_size_, dtype=config.dtype_, device=config.device_),
            config.rms_norm_eps_
        )
        self.post_attention_layernorm_ = GemmaLavaRMSNorm(
            torch.ones(config.hidden_size_, dtype=config.dtype_, device=config.device_),
            config.rms_norm_eps_
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 自注意力
        residual = hidden_states
        hidden_states = self.input_layernorm_(hidden_states)
        hidden_states = self.self_attn_(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rotary_emb=rotary_emb
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states = self.mlp_(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class GemmaLavaRotaryEmbedding(nn.Module):
    def __init__(self, config: GemmaLavaConfig):
        super().__init__()
        self.dim = config.head_dim_
        self.max_seq_len_cached = config.max_seq_len_
        self.base = config.rope_theta_
        
        # 计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim // 2).float() / (self.dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
        
        # 构建缓存
        self._build_cache(self.max_seq_len_cached)
        
    def _build_cache(self, seq_len: int):
        """构建余弦和正弦缓存"""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 注册缓存
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播,返回余弦和正弦缓存"""
        # 获取序列长度
        seq_len = position_ids.max().item() + 1
        
        # 如果需要,扩展缓存
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)
            
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

class GemmaLavaForCausalLM(VLMForCausalLM):
    def __init__(self, config: GemmaLavaConfig) -> None:
        super().__init__()
        self.config_ = config
        
        # 词嵌入层
        self.embed_tokens_ = GemmaLavaEmbedding(
            torch.empty(
                (config.vocab_size_, config.dim_),
                dtype=config.dtype_,
                device=config.device_,
            ),
            config.pad_token_id_,
        )
        
        # 位置编码
        self.rotary_emb_ = GemmaLavaRotaryEmbedding(config)
      
        # Transformer 层
        self.layers_ = nn.ModuleList(
            [GemmaLavaDecoderLayer(config) for i in range(config.n_layers_)]
        )
        
        # 最后的层归一化
        self.norm_ = GemmaLavaRMSNorm(
            torch.ones(config.dim_, dtype=config.dtype_, device=config.device_),
            config.rms_norm_eps_,
        )
        
        # 视觉编码器
        if config.vision_config_ is not None:
            self.vision_model_ = CLIPVisionModel.from_pretrained(
                "C:/Users/hxq11/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41",
                torch_dtype=config.dtype_,
                device_map=config.device_,
                local_files_only=True
            )
            self.vision_model_.requires_grad_(False)  # 冻结视觉编码器
            
            # 多模态投影层
            self.mm_projector_ = nn.Linear(
                self.vision_model_.config.hidden_size,
                config.dim_,
                bias=False,
                dtype=config.dtype_,
                device=config.device_
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[VLMCache]] = None,
        image_features: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        
        # 创建输入参数对象
        input_args = VLMModelInput(
            dtype_=self.config_.dtype_,
            device_=self.config_.device_,
            batch_size_=batch_size,
            seq_length_=seq_length,
            inference_mode_=True
        )
        
        if position_ids is None:
            position_ids = torch.arange(
                seq_length,
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0)
            
        # 1. 获取输入嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. 如果有图像特征，拼接到文本特征前面
        if image_features is not None:
            hidden_states = torch.cat([image_features, hidden_states], dim=1)
            # 更新attention mask
            if attention_mask is not None:
                image_mask = torch.ones(
                    (batch_size, image_features.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # 3. 获取旋转位置编码
        rotary_emb = self.rotary_embed(hidden_states, position_ids)
        
        # 4. 通过解码器层
        for idx, decoder_layer in enumerate(self.layers_):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                input_args,
                rotary_emb,
                attention_mask,
                None,
                layer_past,
            )
            
        # 5. 最后的层归一化
        hidden_states = self.norm(hidden_states)
        
        # 6. 语言模型头部
        logits = self.lm_head_(hidden_states)
        
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """生成文本的方法"""
        # 准备输入
        batch_size = input_ids.shape[0]
        max_length = kwargs.get("max_length", 512)
        # 从配置中获取 pad_token_id
        pad_token_id = kwargs.get("pad_token_id", self.config_.pad_token_id_)
        eos_token_id = kwargs.get("eos_token_id", None)
        
        # 初始化生成序列
        generated_tokens = []
        current_length = 0
        past_key_values = None
        
        # 创建填充掩码，初始时全为1（未填充）
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # 标记填充位置
        if pad_token_id is not None:
            attention_mask = attention_mask.masked_fill(input_ids == pad_token_id, 0)
        
        while current_length < max_length:
            # 前向传播
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                image_features=image_features if current_length == 0 else None,
                **kwargs
            )
            
            # 获取下一个token
            next_token_logits = outputs[:, -1, :]
            # 对填充token的概率进行惩罚
            if pad_token_id is not None:
                next_token_logits[:, pad_token_id] = float('-inf')
                
            next_token = self._get_next_token(
                next_token_logits,
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.0)
            )
            
            # 添加到生成序列
            generated_tokens.append(next_token)
            
            # 检查是否生成了结束符
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
                
            # 更新输入
            input_ids = next_token.unsqueeze(-1)
            current_length += 1
            
            # 更新attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))],
                    dim=-1
                )
        
        # 将生成的token拼接成序列
        return torch.cat(generated_tokens, dim=0).unsqueeze(0)

    def _get_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0
    ) -> torch.Tensor:
        """获取下一个token的辅助方法"""
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
            
        # 应用重复惩罚
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, repetition_penalty)
            
        # 应用top-p采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        # 采样下一个token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """应用重复惩罚"""
        score = torch.where(logits < 0, logits * penalty, logits / penalty)
        return score

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens_(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb_(input_tensor, position_ids)

    def decoder_stack(self) -> List[VLMDecoder]:
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm_(hidden_states)

    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'vision_model_'):
            return None
            
        # 获取视觉模型输出
        vision_outputs = self.vision_model_(image, output_hidden_states=True)
        
        # 获取最后一层的输出
        if hasattr(vision_outputs, 'hidden_states'):
            # 如果有 hidden_states，使用指定层的输出
            selected_layer = self.config_.mm_vision_select_layer_
            hidden_states = vision_outputs.hidden_states[selected_layer]
        else:
            # 如果没有 hidden_states，使用最后一层的输出
            hidden_states = vision_outputs.last_hidden_state
        
        # 根据配置选择特征
        if self.config_.mm_vision_select_feature_ == "patch":
            image_features = hidden_states[:, 1:]  # 除去CLS token
        else:
            image_features = hidden_states[:, [0]]  # 只要CLS token
            
        # 通过投影层
        image_features = self.mm_projector_(image_features)
        return image_features

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda"):
        print(f"Loading configuration from {model_path}")
        config = AutoConfig.from_pretrained(model_path)
        
        # 首先加载预训练模型以获取正确的层数
        print("Loading pretrained model to get configuration...")
        pretrained = GemmaForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 从 LlavaConfig 的 text_config 中获取 Gemma 的配置
        text_config = config.text_config
        
        # 创建 GemmaLavaConfig，使用预训练模型的实际层数
   
        gemma_lava_config = GemmaLavaConfig(
            dim_=text_config.hidden_size,
            n_layers_=len(pretrained.model.layers),  # 从预训练模型获取实际层数
            n_heads_=text_config.num_attention_heads,
            n_kv_heads_=text_config.num_key_value_heads,
            head_dim_=text_config.head_dim,
            hidden_act_=text_config.hidden_act,
            pad_token_id_=text_config.pad_token_id,
            vocab_size_=text_config.vocab_size,
            rms_norm_eps_=getattr(text_config, "rms_norm_eps", 1e-6),
            rope_scaling_=text_config.rope_scaling,
            max_seq_len_=text_config.max_position_embeddings,
            vision_config_=config.vision_config,
            # 从配置中获取相关参数
            mm_vision_select_layer_=config.vision_feature_layer,
            mm_vision_select_feature_=config.vision_feature_select_strategy,
            image_token_len_=config.image_seq_length,
            # 使用配置中的视觉塔路径，如果没有则使用默认值
            mm_vision_tower_=getattr(config, "mm_vision_tower", "openai/clip-vit-large-patch14"),
            # 设置设备和数据类型
            device_=device,
            dtype_=torch.float16 if device == "cuda" else torch.float32
        )
        
        print("Creating GemmaLavaForCausalLM")
        model = cls(gemma_lava_config)
        
        # 加载预训练权重时设置 trust_remote_code=True 和 local_files_only=True
        pretrained = GemmaForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 转换参数名称并复制参数
        state_dict = {}
        
        # 处理语言模型参数
        for name, param in pretrained.state_dict().items():
            if name.startswith("model."):
                # 移除 "model." 前缀
                name = name[6:]
                # 将参数名称转换为我们的格式
                if name == "embed_tokens.weight":
                    state_dict["embed_tokens_.weight"] = param
                elif name == "norm.weight":
                    state_dict["norm_.weight"] = param
                elif name.startswith("layers."):
                    # 处理层参数
                    parts = name.split(".")
                    layer_id = parts[1]
                    if parts[2] == "self_attn":
                        # 处理注意力层参数
                        proj_type = parts[3]  # q_proj, k_proj, v_proj, o_proj
                        state_dict[f"layers_.{layer_id}.self_attn_.{proj_type}_"] = param
                    elif parts[2] == "mlp":
                        # 处理 MLP 层参数
                        proj_type = parts[3]  # gate_proj, up_proj, down_proj
                        state_dict[f"layers_.{layer_id}.mlp_.{proj_type}_"] = param
                    elif parts[2].endswith("layernorm"):
                        # 处理层归一化参数
                        norm_type = parts[2]  # input_layernorm, post_attention_layernorm
                        state_dict[f"layers_.{layer_id}.{norm_type}_"] = param
        
        # 加载视觉模型
        print("Loading CLIP vision model from local cache...")
        vision_model = CLIPVisionModel.from_pretrained(
            "C:/Users/hxq11/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True  # 强制使用本地文件
        )
        
        # 添加视觉模型参数
        for name, param in vision_model.state_dict().items():
            # 将视觉模型参数添加到状态字典中
            state_dict[f"vision_model_.{name}"] = param
            
        # 初始化多模态投影层
        print("Initializing multi-modal projector...")
        projector = nn.Linear(
            vision_model.config.hidden_size,
            gemma_lava_config.dim_,
            bias=False,
            dtype=gemma_lava_config.dtype_,
            device=gemma_lava_config.device_
        )
        
        # 添加投影层参数
        state_dict["mm_projector_.weight"] = projector.weight.data
        
        # 加载参数
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # 只打印关键信息
        if len(missing_keys) > 0:
            print(f"\nMissing keys: {len(missing_keys)}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # 释放原始模型内存
        del pretrained
        del vision_model
        torch.cuda.empty_cache()
        
        return model