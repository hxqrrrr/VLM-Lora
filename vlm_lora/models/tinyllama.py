import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlm_lora.common import (
    VLMModelConfig,
    VLMCache,
    VLMDecoder,
    VLMForCausalLM,
)


@dataclass
class LlamaConfig(VLMModelConfig):
    # 模型基础配置
    hidden_size_: int = 2048
    num_attention_heads_: int = 32
    num_hidden_layers_: int = 22
    max_seq_len_: int = 2048
    vocab_size_: int = 32000
    
    # 模型特定配置
    head_dim_: int = 64  # hidden_size_ // num_attention_heads_
    intermediate_size_: int = 5632
    rope_theta_: float = 10000
    rms_norm_eps_: float = 1e-6


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码
    Args:
        q: shape (batch_size, num_heads, seq_length, head_dim)
        k: shape (batch_size, num_heads, seq_length, head_dim)
        cos: shape (batch_size, num_heads, seq_length, head_dim)
        sin: shape (batch_size, num_heads, seq_length, head_dim)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.dim = config.head_dim_
        self.max_position_embeddings = config.max_seq_len_
        self.base = config.rope_theta_
        
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim // 2).float() / (self.dim // 2)))
        inv_freq = inv_freq.to(position_ids.device)
        
        # 计算 sin/cos 值
        t = position_ids.float().unsqueeze(-1) @ inv_freq.unsqueeze(0)  # [batch_size, seq_length, dim/2]
        
        # 计算 sin/cos
        sin = torch.sin(t)  # [batch_size, seq_length, dim/2]
        cos = torch.cos(t)  # [batch_size, seq_length, dim/2]
        
        # 扩展维度
        sin = torch.cat([sin, sin], dim=-1)  # [batch_size, seq_length, dim]
        cos = torch.cat([cos, cos], dim=-1)  # [batch_size, seq_length, dim]
        
        # 扩展到所有注意力头
        sin = sin.unsqueeze(1)  # [batch_size, 1, seq_length, dim]
        cos = cos.unsqueeze(1)  # [batch_size, 1, seq_length, dim]
        
        return cos, sin


class LlamaEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight_ = nn.Parameter(torch.ones(hidden_size))
        self.norm_eps_ = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.norm_eps_)
        return (self.weight_ * hidden_states).to(input_dtype)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size_ = config.hidden_size_
        self.num_heads_ = config.num_attention_heads_
        self.head_dim_ = config.head_dim_
        
        self.q_proj_ = nn.Linear(self.hidden_size_, self.num_heads_ * self.head_dim_, bias=False)
        self.k_proj_ = nn.Linear(self.hidden_size_, self.num_heads_ * self.head_dim_, bias=False)
        self.v_proj_ = nn.Linear(self.hidden_size_, self.num_heads_ * self.head_dim_, bias=False)
        self.o_proj_ = nn.Linear(self.num_heads_ * self.head_dim_, self.hidden_size_, bias=False)
        
        self.rotary_emb_ = None  # 将在forward时设置
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
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
        
        # 应用 RoPE 位置编码
        if self.rotary_emb_ is not None and position_ids is not None:
            cos, sin = self.rotary_emb_(position_ids)
            # cos 和 sin 已经是 [batch_size, num_heads, head_dim] 的形状
            # 扩展序列维度
            cos = cos.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
            sin = sin.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim_)
        
        if attention_mask is not None:
            # 确保 attention_mask 的维度正确
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value)
        
        # 重塑并投影输出
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_length, -1)
        hidden_states = self.o_proj_(hidden_states)
        
        return hidden_states


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj_ = nn.Linear(config.hidden_size_, config.intermediate_size_, bias=False)
        self.up_proj_ = nn.Linear(config.hidden_size_, config.intermediate_size_, bias=False)
        self.down_proj_ = nn.Linear(config.intermediate_size_, config.hidden_size_, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj_(x))
        up = self.up_proj_(x)
        return self.down_proj_(gate * up)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn_ = LlamaAttention(config)
        self.mlp_ = LlamaMLP(config)
        self.input_norm_ = LlamaRMSNorm(config.hidden_size_, config.rms_norm_eps_)
        self.post_attn_norm_ = LlamaRMSNorm(config.hidden_size_, config.rms_norm_eps_)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 自注意力
        residual = hidden_states
        hidden_states = self.input_norm_(hidden_states)
        hidden_states = self.self_attn_(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attn_norm_(hidden_states)
        hidden_states = self.mlp_(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class TinyLLaMAForCausalLM(VLMForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config_ = config
        
        # 加载预训练模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path_,
            device_map=config.device_,
            torch_dtype=config.dtype_,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path_,
            trust_remote_code=True,
        )
        
        # 保存配置到实例变量
        self.vocab_size_ = self.model.config.vocab_size
        self.hidden_size_ = self.model.config.hidden_size
        self.num_attention_heads_ = self.model.config.num_attention_heads
        self.num_hidden_layers_ = self.model.config.num_hidden_layers
        self.pad_token_id_ = getattr(self.model.config, "pad_token_id", 0)
        
        # 初始化组件
        self.embed_tokens_ = LlamaEmbedding(
            self.model.model.embed_tokens.weight,
            self.pad_token_id_
        )
        self.layers_ = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers_)
        ])
        self.norm_ = LlamaRMSNorm(
            config.hidden_size_,
            config.rms_norm_eps_
        )
        self.rotary_emb_ = LlamaRotaryEmbedding(config)
        self.lm_head_ = nn.Linear(config.hidden_size_, config.vocab_size_, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 获取词嵌入
        hidden_states = self.embed_tokens_(input_ids)
        
        # 生成位置编码的位置 ID
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 通过所有解码器层
        for layer in self.layers_:
            layer.self_attn_.rotary_emb_ = self.rotary_emb_
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
        # 最终归一化
        hidden_states = self.norm_(hidden_states)
        
        # 语言模型头
        logits = self.lm_head_(hidden_states)
        
        return logits

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入的 token ID 转换为词嵌入"""
        return self.embed_tokens_(input_ids)

    def decoder_stack(self) -> List[VLMDecoder]:
        """返回解码器层的列表"""
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用最终的层归一化"""
        return self.norm_(hidden_states)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "TinyLLaMAForCausalLM":
        """从预训练模型创建实例"""
        config = LlamaConfig(
            name_or_path_=model_path,
            device_=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            dtype_=kwargs.get("dtype", torch.float32)
        )
        return cls(config)
