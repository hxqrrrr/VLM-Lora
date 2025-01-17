import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def default_rope_init(
    config,
    device,
    dim: int,
    base: float = 10000,
    factor: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """默认的 RoPE 初始化函数。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    return inv_freq, 1.0


def dynamic_rope_init(
    config,
    device,
    dim: int,
    base: float = 10000,
    factor: float = 1.0,
    seq_len: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """动态 RoPE 初始化函数。"""
    if seq_len is None:
        seq_len = config.max_seq_len_
    
    # 计算缩放因子
    scale = (seq_len / config.max_seq_len_) ** (dim / (dim - 2))
    inv_freq = 1.0 / ((base * scale) ** (torch.arange(0, dim, 2).float().to(device) / dim))
    return inv_freq, 1.0


def linear_rope_init(
    config,
    device,
    dim: int,
    base: float = 10000,
    factor: float = 1.0,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """线性 RoPE 初始化函数。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    return inv_freq, factor


ROPE_INIT_FUNCTIONS = {
    "default": default_rope_init,
    "dynamic": dynamic_rope_init,
    "linear": linear_rope_init,
}


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """使用传统的注意力计算方法。

    Args:
        query: 查询张量 [batch_size, n_heads, seq_len, head_dim]
        key: 键张量 [batch_size, n_heads, seq_len, head_dim]
        value: 值张量 [batch_size, n_heads, seq_len, head_dim]
        attention_mask: 注意力掩码 [batch_size, 1, seq_len, seq_len]

    Returns:
        注意力输出 [batch_size, seq_len, n_heads * head_dim]
    """
    # 计算注意力分数
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    
    # 应用注意力掩码
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    # 应用 softmax
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # 计算注意力输出
    return torch.matmul(attn_weights, value)

