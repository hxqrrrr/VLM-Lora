from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
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


@dataclass
class LlamaConfig(VLMModelConfig):
    rms_norm_eps_: float = 1e-6
    rope_scaling_: Optional[Dict[str, Any]] = None


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


class LlamaAttention(VLMAttention):
    def __init__(
        self,
        wq: nn.Module,
        wk: nn.Module,
        wv: nn.Module,
        wo: nn.Module,
        idx: int,
        args: LlamaConfig,
    ):
        super().__init__()
        # attention
        self.wq_: Linear = Linear(wq, args.device_)
        self.wk_: Linear = Linear(wk, args.device_)
        self.wv_: Linear = Linear(wv, args.device_)
        self.wo_: Linear = Linear(wo, args.device_)
        # config
        self.layer_idx_ = idx
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.head_dim_
        self.dtype_ = args.dtype_
        self.is_causal_ = True

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "q_proj": self.wq_,
            "k_proj": self.wk_,
            "v_proj": self.wv_,
            "o_proj": self.wo_,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: VLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[VLMCache] = None,
    ):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_).transpose(1, 2)

        cos, sin = rotary_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            xk, xv = past_key_value.update(xk, xv, self.layer_idx_, cache_kwargs)

        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = eager_attention_forward(xq, xk, xv, attention_mask)
        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)

        return self.wo_.forward(attention_score, input_args)


class LlamaMLP(VLMFeedForward):
    def __init__(
        self, w1: nn.Module, w2: nn.Module, w3: nn.Module, args: LlamaConfig
    ) -> None:
        super().__init__()
        self.w1_: Linear = Linear(w1, args.device_)
        self.w2_: Linear = Linear(w2, args.device_)
        self.w3_: Linear = Linear(w3, args.device_)
        self.act_ = ACT2FN[args.hidden_act_]

    def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "gate_proj": self.w1_,
            "down_proj": self.w2_,
            "up_proj": self.w3_,
        }

    def forward(
        self, data: torch.Tensor, input_args: VLMModelInput
    ) -> torch.Tensor:
        w1 = self.w1_.forward(data, input_args)
        w3 = self.w3_.forward(data, input_args)
        return self.w2_.forward(self.act_(w1) * w3, input_args)


class LlamaDecoderLayer(VLMDecoder):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_: LlamaAttention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: LlamaRMSNorm = None
        self.post_attention_layernorm_: LlamaRMSNorm = None

    def state_dict(self) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module]]:
        return self.self_attn_.state_dict(), self.mlp_.state_dict()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: VLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[VLMCache] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm_(hidden_states)
        hidden_states = self.self_attn_.forward(
            hidden_states,
            input_args,
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states = self.mlp_.forward(hidden_states, input_args)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaForCausalLM(VLMForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: LlamaEmbedding = None
        self.norm_: LlamaRMSNorm = None
        self.rotary_emb_ = LlamaRotaryEmbedding(config)
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.layers_: List[LlamaDecoderLayer] = []

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

    def causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[VLMCache],
    ) -> torch.Tensor:
        return prepare_4d_causal_attention_mask(
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
        )

    def model_config(self) -> LlamaConfig:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model: modeling_llama.LlamaForCausalLM,
        device: str = executor.default_device_name(),
    ):
        llm_config: modeling_llama.LlamaConfig = llm_model.config
        llm_args = LlamaConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.hidden_size // llm_config.num_attention_heads,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=llm_config.hidden_act,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            rope_scaling_=llm_config.rope_scaling,
            pad_token_id_=llm_config.pad_token_id,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = LlamaForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = LlamaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = LlamaAttention(
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(
                LlamaMLP(
                    layer.mlp.gate_proj,
                    layer.mlp.down_proj,
                    layer.mlp.up_proj,
                    llm_args,
                )
            )
            decoder.input_layernorm_ = LlamaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = LlamaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_
            )
            model.layers_.append(decoder)

        return model


def prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, ...],
    cache_position: Optional[torch.Tensor],
    past_key_values: Optional[VLMCache],
) -> Optional[torch.Tensor]:
    """准备4D的因果注意力掩码。

    Args:
        attention_mask: 输入的注意力掩码
        input_shape: 输入张量的形状
        cache_position: 缓存位置
        past_key_values: 过去的键值对

    Returns:
        4D的注意力掩码
    """
    batch_size, seq_length = input_shape

    if attention_mask is not None:
        # 确保注意力掩码是4D的
        if attention_mask.dim() <= 2:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
        
        # 扩展注意力掩码
        attention_mask = attention_mask.to(dtype=torch.bool)
        attention_mask = ~attention_mask

    # 创建因果掩码
    seq_ids = torch.arange(seq_length, device=attention_mask.device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
    
    if cache_position is not None:
        max_seq_len = cache_position.max().item() + seq_length
        causal_mask = causal_mask + cache_position[:, None, None]
    else:
        max_seq_len = seq_length
    
    causal_mask = causal_mask <= seq_ids[None, :, None]
    causal_mask = causal_mask.to(attention_mask.device)
    causal_mask = causal_mask[:, None, :, :]

    if attention_mask is not None:
        attention_mask = attention_mask | ~causal_mask
    else:
        attention_mask = ~causal_mask

    return attention_mask.to(dtype=torch.float32)
