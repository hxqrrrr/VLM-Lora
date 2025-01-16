import copy
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import snapshot_download

from vlm_lora.common import (
    CHECKPOINT_CLASSES,
    AdapterConfig,
    Linear,
    VLMCache,
    VLMDecoder,
    VLMForCausalLM,
    VLMModelConfig,
    VLMModelInput,
    VLMModelOutput,
    VLMOutput,
    LoraConfig,
)

class VLMModel(torch.nn.Module):
    def __init__(self, model: VLMForCausalLM):
        super().__init__()
        args: VLMModelConfig = model.config_
        if args.vocab_size_ >= torch.finfo(args.dtype_).max:
            logging.warn(
                f"vocab_size >= max({args.dtype_}), consider load model with higher precision."
            )
        self.model_ = model
        self.config_ = args
        # configs
        self.name_or_path_ = args.name_or_path_
        self.vocab_size_ = args.vocab_size_
        self.device_ = args.device_
        self.dtype_ = args.dtype_

        self.output_ = OutputLayer()
        # adapter configs
        self.adapter_configs_: Dict[str, AdapterConfig] = {}

    def _prepare_inputs(
        self, input_args: VLMModelInput, past_key_values: Optional[VLMCache] = None
    ):
        """准备模型输入"""
        assert input_args.batch_tokens_ is not None, "Model have no input."
        assert (
            input_args.gradient_checkpoint_ == "none" or past_key_values is None
        ), "Cache is incompatible with gradient checkpointing."
        assert (
            not input_args.inference_mode_ or input_args.gradient_checkpoint_ == "none"
        ), "Can not use gradient checkpoint when inference."

        # prepare inputs
        if isinstance(input_args.batch_tokens_, torch.Tensor):
            input_ids = input_args.batch_tokens_.to(
                dtype=torch.int64, device=self.device_
            )
        else:
            input_ids = torch.tensor(
                input_args.batch_tokens_, dtype=torch.int64, device=self.device_
            )

        inputs_embeds = self.model_.embed_tokens(input_ids)
        if input_args.gradient_checkpoint_ != "none":
            inputs_embeds.requires_grad_(True)

        # prepare cache
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        if past_seen_tokens is None:
            past_seen_tokens = 0

        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        # prepare mask
        if input_args.batch_masks_ is not None:
            # 2d mask is passed through the layers
            if isinstance(input_args.batch_masks_, torch.Tensor):
                attention_mask = input_args.batch_masks_.to(
                    dtype=torch.int64, device=self.device_
                )
            else:
                attention_mask = torch.tensor(
                    input_args.batch_masks_, dtype=torch.int64, device=self.device_
                )
        else:
            attention_mask = None

        if self.config_.attn_implementation_ != "flash_attn":
            causal_mask = self.model_.causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values
            )
        else:
            causal_mask = attention_mask

        return input_ids, inputs_embeds, attention_mask, causal_mask, cache_position

    def _call_decoder_stack(
        self,
        hidden_states: torch.Tensor,
        input_args: VLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[VLMCache] = None,
    ):
        """调用解码器堆栈"""
        # decoder layers
        num_adapters = len(input_args.batch_configs_)
        all_router_logits = [[] for _ in range(num_adapters)]
        gradient_checkpoint = CHECKPOINT_CLASSES[input_args.gradient_checkpoint_]

        for decoder_layer in self.model_.decoder_stack():
            hidden_states, *router_logits = gradient_checkpoint(
                decoder_layer.forward,
                hidden_states,
                input_args,
                rotary_emb,
                attention_mask,
                cache_position,
                past_key_value,
            )
            if len(router_logits) == 0:
                continue
            # collecting router logits
            assert len(router_logits) == num_adapters
            for idx in range(num_adapters):
                if router_logits[idx] is not None:
                    all_router_logits[idx].append(router_logits[idx])

        hidden_states = self.model_.norm(hidden_states)

        return hidden_states, all_router_logits

    def forward(
        self, input_args: VLMModelInput, past_key_values: Optional[VLMCache] = None
    ) -> List[VLMModelOutput]:
        """前向传播"""
        input_ids, inputs_embeds, attention_mask, causal_mask, cache_position = (
            self._prepare_inputs(input_args, past_key_values)
        )

        labels = input_args.batch_labels_

        input_args.batch_labels_ = None
        input_args.batch_tokens_ = None
        input_args.batch_masks_ = None

        # embed positions
        hidden_states = inputs_embeds

        rotary_emb = self.model_.rotary_embed(
            hidden_states, cache_position.unsqueeze(0)
        )

        hidden_states, all_router_logits = self._call_decoder_stack(
            hidden_states,
            input_args,
            rotary_emb,
            causal_mask,
            cache_position,
            past_key_values,
        )

        # calculate loss
        output = self.output_(hidden_states, input_args)
        assert isinstance(output, List)
        for idx, lora_config in enumerate(input_args.batch_configs_):
            output_data = output[idx]
            assert isinstance(output_data, VLMModelOutput)
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            output_data.batch_start_idx_ = start_idx
            output_data.batch_end_idx_ = end_idx
            if input_args.output_router_logits_ and len(all_router_logits[idx]) > 0:
                output_data.router_logits = torch.stack(all_router_logits[idx])
            if labels is None:
                continue
            # compute loss when labels provided
            output_data.loss = output_data.loss_fn_(
                input_ids[start_idx:end_idx],
                output_data.logits,
                labels[start_idx:end_idx],
            )
            output_data.loss_fn_ = None

        return output

    def add_adapter(self, config: AdapterConfig) -> str:
        """添加适配器"""
        adapter_name = config.adapter_name
        assert adapter_name not in self.adapter_configs_, "adapter already exists"
        self.adapter_configs_[adapter_name] = config

        # 初始化输出层
        if config.task_name == "casual":
            output_layer = CasualOutputLayer(
                vocab_size=self.vocab_size_,
                weight=self.model_.lm_head_
            )
        else:
            raise ValueError(f"unknown task type {config.task_type}")

        self.output_.layers_[adapter_name] = output_layer

        return adapter_name

class OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_: Dict[str, torch.nn.Module] = {}

    def forward(
        self, data: torch.Tensor, input_args: VLMModelInput
    ) -> List[VLMModelOutput]:
        outputs = []
        for lora_config in input_args.batch_configs_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            assert adapter_name != "" and adapter_name in self.layers_
            layer = self.layers_[adapter_name]
            outputs.append(
                VLMModelOutput(
                    adapter_name=adapter_name,
                    logits=layer.forward(data[start_idx:end_idx]),
                    loss_fn_=layer.loss,
                )
            )

        return outputs 

class CasualOutputLayer(torch.nn.Module):
    """用于语言模型的输出层"""
    def __init__(self, vocab_size: int, weight: torch.nn.Linear):
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = weight
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算 logits"""
        return self.weight(hidden_states)
        
    def loss(self, input_ids: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss 