import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import urllib3


from vlm_lora.common import (
    VLMModelConfig,
    VLMCache,
    VLMDecoder,
    VLMForCausalLM,
)
from vlm_lora.models.base import VLMModel


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, scaling_factor=1.0, rope_type="default"):
        super().__init__()
        self.dim = config.head_dim_ if config else 64
        self.max_position_embeddings = config.max_seq_len_ if config else 2048
        self.base = config.rope_theta_ if config else 10000
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type

    def forward(self, x, position_ids):
        # 计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        inv_freq = inv_freq.to(x.device)
        
        # 计算 sin/cos 值
        t = position_ids.float().unsqueeze(-1) @ inv_freq.unsqueeze(0)
        
        # 扩展维度以匹配注意力头
        sin = torch.cat([torch.sin(t), torch.sin(t)], dim=-1)
        cos = torch.cat([torch.cos(t), torch.cos(t)], dim=-1)
        
        # 添加必要的维度
        sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
        
        return cos, sin


class LlamaEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)
        return data


class LlamaRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)
        return (self.weight_ * data).to(input_dtype)


class TinyLLaMAForCausalLM(VLMForCausalLM):
    def __init__(self, config: VLMModelConfig):
        super().__init__()
        self.config_ = config
        
        # 加载预训练模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path_,
            device_map=config.device_,
            torch_dtype=config.dtype_,
            trust_remote_code=True,
            token=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path_,
            trust_remote_code=True,
            token=True
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
        self.norm_ = LlamaRMSNorm(
            self.model.model.norm.weight,
            self.model.config.rms_norm_eps
        )
        self.rotary_emb_ = LlamaRotaryEmbedding(
            config=None,
            scaling_factor=1.0,
            rope_type="default"
        )
        self.lm_head_ = self.model.lm_head
        self.layers_ = self.model.model.layers

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播，完全手动实现处理流程"""
        # 获取输入的形状信息
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 生成位置编码的位置 ID
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 获取 RoPE 位置编码
        cos, sin = self.rotary_embed(hidden_states, position_ids)
        
        # 通过解码器层
        for layer in self.decoder_stack():
            # 设置 RoPE 编码
            layer.self_attn.rotary_emb = lambda *args, **kwargs: (cos, sin)
            # 前向传播
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        
        # 最终的归一化
        hidden_states = self.norm(hidden_states)
        
        # 语言模型头
        logits = self.lm_head_(hidden_states)
        
        return logits

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入的 token ID 转换为词嵌入"""
        return self.embed_tokens_(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 RoPE 位置编码"""
        return self.rotary_emb_(input_tensor, position_ids)

    def decoder_stack(self) -> List[VLMDecoder]:
        """返回解码器层的列表"""
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用最终的层归一化"""
        return self.norm_(hidden_states)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "TinyLLaMAForCausalLM":
        """从预训练模型创建实例"""
        config = VLMModelConfig(
            name_or_path_=model_path,
            device_=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            dtype_=kwargs.get("dtype", torch.float32)
        )
        return cls(config)
