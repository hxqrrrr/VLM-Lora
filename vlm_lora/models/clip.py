from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip import modeling_clip

from vlm_lora.common import (
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
class CLIPConfig(VLMModelConfig):
    # 视觉编码器配置
    vision_embed_dim_: int = 768
    vision_patch_size_: int = 32
    vision_layers_: int = 12
    vision_heads_: int = 12
    vision_width_: int = 768
    vision_dropout_: float = 0.0
    image_size_: int = 224
    
    # 文本编码器配置
    text_embed_dim_: int = 512
    text_layers_: int = 12
    text_heads_: int = 8
    text_width_: int = 512
    text_dropout_: float = 0.0
    max_position_embeddings_: int = 77
    
    # 投影头配置
    projection_dim_: int = 512


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config_ = config
        self.embed_dim_ = config.vision_embed_dim_
        self.image_size_ = config.image_size_
        self.patch_size_ = config.vision_patch_size_

        self.patch_embedding_ = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim_,
            kernel_size=self.patch_size_,
            stride=self.patch_size_,
            bias=False,
        )

        self.num_patches_ = (self.image_size_ // self.patch_size_) ** 2
        self.num_positions_ = self.num_patches_ + 1
        self.position_embedding_ = nn.Embedding(self.num_positions_, self.embed_dim_)
        self.register_buffer("position_ids", torch.arange(self.num_positions_).expand((1, -1)))

        self.cls_embedding_ = nn.Parameter(torch.randn(1, 1, self.embed_dim_))
        self.dropout_ = nn.Dropout(config.vision_dropout_)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding_(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.cls_embedding_.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding_(self.position_ids)
        embeddings = self.dropout_(embeddings)

        return embeddings


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config_ = config
        self.embed_dim_ = config.text_embed_dim_
        
        self.token_embedding_ = nn.Embedding(config.vocab_size_, self.embed_dim_)
        self.position_embedding_ = nn.Embedding(config.max_position_embeddings_, self.embed_dim_)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings_).expand((1, -1)),
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length]

        embeddings = self.token_embedding_(input_ids)
        embeddings = embeddings + self.position_embedding_(position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPConfig, is_cross_attention: bool = False):
        super().__init__()
        self.config_ = config
        self.embed_dim_ = config.vision_embed_dim_ if not is_cross_attention else config.text_embed_dim_
        self.num_heads_ = config.vision_heads_ if not is_cross_attention else config.text_heads_
        self.head_dim_ = self.embed_dim_ // self.num_heads_
        self.scale_ = self.head_dim_ ** -0.5

        self.k_proj_ = nn.Linear(self.embed_dim_, self.embed_dim_)
        self.v_proj_ = nn.Linear(self.embed_dim_, self.embed_dim_)
        self.q_proj_ = nn.Linear(self.embed_dim_, self.embed_dim_)
        self.out_proj_ = nn.Linear(self.embed_dim_, self.embed_dim_)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[:2]

        query = self.q_proj_(hidden_states)
        key = self.k_proj_(hidden_states)
        value = self.v_proj_(hidden_states)

        query = query.view(batch_size, -1, self.num_heads_, self.head_dim_).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads_, self.head_dim_).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads_, self.head_dim_).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale_

        if causal_attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                causal_attention_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attention_probs = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.embed_dim_)
        hidden_states = self.out_proj_(hidden_states)

        if output_attentions:
            return hidden_states, attention_probs

        return hidden_states, None


class CLIPMLP(VLMFeedForward):
    def __init__(self, config: CLIPConfig, is_text: bool = False):
        super().__init__()
        self.config_ = config
        self.embed_dim_ = config.text_embed_dim_ if is_text else config.vision_embed_dim_
        self.intermediate_size_ = config.text_width_ if is_text else config.vision_width_

        self.activation_fn_ = nn.GELU()
        self.fc1_ = nn.Linear(self.embed_dim_, self.intermediate_size_)
        self.fc2_ = nn.Linear(self.intermediate_size_, self.embed_dim_)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1_(hidden_states)
        hidden_states = self.activation_fn_(hidden_states)
        hidden_states = self.fc2_(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig, is_text: bool = False):
        super().__init__()
        self.embed_dim_ = config.text_embed_dim_ if is_text else config.vision_embed_dim_
        
        self.self_attn_ = CLIPAttention(config, is_cross_attention=is_text)
        self.layer_norm1_ = nn.LayerNorm(self.embed_dim_)
        self.mlp_ = CLIPMLP(config, is_text=is_text)
        self.layer_norm2_ = nn.LayerNorm(self.embed_dim_)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states

        hidden_states = self.layer_norm1_(hidden_states)
        hidden_states, attn_weights = self.self_attn_(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2_(hidden_states)
        hidden_states = self.mlp_(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, attn_weights

        return hidden_states, None


class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPConfig, is_text: bool = False):
        super().__init__()
        self.config_ = config
        self.layers_ = nn.ModuleList(
            [CLIPEncoderLayer(config, is_text=is_text) for _ in range(
                config.text_layers_ if is_text else config.vision_layers_
            )]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Dict]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers_:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions,
            )

            if output_attentions and attn_weights is not None:
                all_attentions += (attn_weights,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
            }

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config_ = config
        self.embeddings_ = CLIPVisionEmbeddings(config)
        self.encoder_ = CLIPEncoder(config)
        self.post_layernorm_ = nn.LayerNorm(config.vision_embed_dim_)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Dict]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings_(pixel_values)
        encoder_outputs = self.encoder_(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_outputs, dict):
            last_hidden_state = encoder_outputs["last_hidden_state"]
        else:
            last_hidden_state = encoder_outputs[0]

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm_(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "last_hidden_state": last_hidden_state,
            "pooled_output": pooled_output,
            "hidden_states": encoder_outputs.get("hidden_states"),
            "attentions": encoder_outputs.get("attentions"),
        }


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config_ = config
        self.embeddings_ = CLIPTextEmbeddings(config)
        self.encoder_ = CLIPEncoder(config, is_text=True)
        self.final_layer_norm_ = nn.LayerNorm(config.text_embed_dim_)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Dict]:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        hidden_states = self.embeddings_(input_ids)

        # CLIP's text model uses causal mask, prepare it here.
        batch_size, seq_length = input_ids.shape
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length)
        causal_attention_mask = causal_attention_mask.to(hidden_states.device)

        encoder_outputs = self.encoder_(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_outputs, dict):
            last_hidden_state = encoder_outputs["last_hidden_state"]
        else:
            last_hidden_state = encoder_outputs[0]

        last_hidden_state = self.final_layer_norm_(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "last_hidden_state": last_hidden_state,
            "pooled_output": pooled_output,
            "hidden_states": encoder_outputs.get("hidden_states"),
            "attentions": encoder_outputs.get("attentions"),
        }

    def _build_causal_attention_mask(self, batch_size: int, seq_length: int) -> torch.Tensor:
        mask = torch.empty(batch_size, seq_length, seq_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config_ = config

        self.vision_model_ = CLIPVisionTransformer(config)
        self.text_model_ = CLIPTextTransformer(config)

        self.visual_projection_ = nn.Linear(config.vision_embed_dim_, config.projection_dim_, bias=False)
        self.text_projection_ = nn.Linear(config.text_embed_dim_, config.projection_dim_, bias=False)
        self.logit_scale_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_text_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        text_outputs = self.text_model_(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs["pooled_output"] if isinstance(text_outputs, dict) else text_outputs[1]
        text_features = self.text_projection_(pooled_output)
        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        vision_outputs = self.vision_model_(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs["pooled_output"] if isinstance(vision_outputs, dict) else vision_outputs[1]
        image_features = self.visual_projection_(pooled_output)
        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict]:
        vision_outputs = self.vision_model_(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model_(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs["pooled_output"] if isinstance(vision_outputs, dict) else vision_outputs[1]
        text_embeds = text_outputs["pooled_output"] if isinstance(text_outputs, dict) else text_outputs[1]

        # normalized features
        image_features = self.visual_projection_(image_embeds)
        text_features = self.text_projection_(text_embeds)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale_.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "text_embeds": text_embeds,
            "image_embeds": image_embeds,
            "text_model_output": text_outputs,
            "vision_model_output": vision_outputs,
        }

    @staticmethod
    def from_pretrained(
        clip_model: modeling_clip.CLIPModel,
        device: str = executor.default_device_name(),
    ):
        clip_config = clip_model.config
        config = CLIPConfig(
            vision_embed_dim_=clip_config.vision_config.hidden_size,
            vision_patch_size_=clip_config.vision_config.patch_size,
            vision_layers_=clip_config.vision_config.num_hidden_layers,
            vision_heads_=clip_config.vision_config.num_attention_heads,
            vision_width_=clip_config.vision_config.intermediate_size,
            vision_dropout_=clip_config.vision_config.dropout,
            image_size_=clip_config.vision_config.image_size,
            text_embed_dim_=clip_config.text_config.hidden_size,
            text_layers_=clip_config.text_config.num_hidden_layers,
            text_heads_=clip_config.text_config.num_attention_heads,
            text_width_=clip_config.text_config.intermediate_size,
            text_dropout_=clip_config.text_config.dropout,
            max_position_embeddings_=clip_config.text_config.max_position_embeddings,
            projection_dim_=clip_config.projection_dim,
            vocab_size_=clip_config.text_config.vocab_size,
            device_=torch.device(device),
        )

        model = CLIPModel(config)
        clip_model.requires_grad_(False)

        # 复制视觉模型参数
        copy_parameters(clip_model.vision_model.embeddings.patch_embedding, model.vision_model_.embeddings_.patch_embedding_)
        copy_parameters(clip_model.vision_model.embeddings.position_embedding, model.vision_model_.embeddings_.position_embedding_)
        model.vision_model_.embeddings_.cls_embedding_.data.copy_(clip_model.vision_model.embeddings.class_embedding)

        for src_layer, tgt_layer in zip(clip_model.vision_model.encoder.layers, model.vision_model_.encoder_.layers_):
            copy_parameters(src_layer.self_attn.k_proj, tgt_layer.self_attn_.k_proj_)
            copy_parameters(src_layer.self_attn.v_proj, tgt_layer.self_attn_.v_proj_)
            copy_parameters(src_layer.self_attn.q_proj, tgt_layer.self_attn_.q_proj_)
            copy_parameters(src_layer.self_attn.out_proj, tgt_layer.self_attn_.out_proj_)
            copy_parameters(src_layer.layer_norm1, tgt_layer.layer_norm1_)
            copy_parameters(src_layer.mlp.fc1, tgt_layer.mlp_.fc1_)
            copy_parameters(src_layer.mlp.fc2, tgt_layer.mlp_.fc2_)
            copy_parameters(src_layer.layer_norm2, tgt_layer.layer_norm2_)

        copy_parameters(clip_model.vision_model.post_layernorm, model.vision_model_.post_layernorm_)

        # 复制文本模型参数
        copy_parameters(clip_model.text_model.embeddings.token_embedding, model.text_model_.embeddings_.token_embedding_)
        copy_parameters(clip_model.text_model.embeddings.position_embedding, model.text_model_.embeddings_.position_embedding_)

        for src_layer, tgt_layer in zip(clip_model.text_model.encoder.layers, model.text_model_.encoder_.layers_):
            copy_parameters(src_layer.self_attn.k_proj, tgt_layer.self_attn_.k_proj_)
            copy_parameters(src_layer.self_attn.v_proj, tgt_layer.self_attn_.v_proj_)
            copy_parameters(src_layer.self_attn.q_proj, tgt_layer.self_attn_.q_proj_)
            copy_parameters(src_layer.self_attn.out_proj, tgt_layer.self_attn_.out_proj_)
            copy_parameters(src_layer.layer_norm1, tgt_layer.layer_norm1_)
            copy_parameters(src_layer.mlp.fc1, tgt_layer.mlp_.fc1_)
            copy_parameters(src_layer.mlp.fc2, tgt_layer.mlp_.fc2_)
            copy_parameters(src_layer.layer_norm2, tgt_layer.layer_norm2_)

        copy_parameters(clip_model.text_model.final_layer_norm, model.text_model_.final_layer_norm_)

        # 复制投影层参数
        copy_parameters(clip_model.visual_projection, model.visual_projection_)
        copy_parameters(clip_model.text_projection, model.text_projection_)
        model.logit_scale_.data.copy_(clip_model.logit_scale)

        return model


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = F.cross_entropy(similarity, torch.arange(similarity.shape[0], device=similarity.device))
    image_loss = F.cross_entropy(similarity.t(), torch.arange(similarity.shape[0], device=similarity.device))
    return (caption_loss + image_loss) / 2.0
