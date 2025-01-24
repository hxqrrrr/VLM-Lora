import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, LlamaForCausalLM, LlamaTokenizer

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
    model_type_: str = "gemma"  # 模型类型: "gemma" 或 "llama"
    hidden_size_: int = 2048    # Gemma-2b 的隐藏层大小
    num_attention_heads_: int = 8
    num_hidden_layers_: int = 18
    max_seq_len_: int = 8192    # Gemma 支持更长序列
    vocab_size_: int = 256000   # Gemma 词表大小
    
    # Gemma 特有配置
    num_key_value_heads_: int = 1  # GQA头数
    head_dim_: int = 256           # 每个头的维度
    hidden_act_: str = "gelu"      # 激活函数
    
    # 视觉相关配置
    vision_tower_: str = "openai/clip-vit-large-patch14"
    mm_vision_select_layer_: int = -2
    mm_vision_select_feature_: str = "patch"
    image_aspect_ratio_: str = "square"
    tune_mm_mlp_adapter_: bool = False
    mm_use_im_start_end_: bool = False
    mm_hidden_size_: int = 1024
    
    # 设备和数据类型配置
    dtype_: torch.dtype = torch.float16
    device_map_: Optional[str] = None

    def __post_init__(self):
        """初始化后的处理"""
        if self.model_type_ == "gemma":
            # Gemma特有的配置检查和调整
            assert self.num_key_value_heads_ <= self.num_attention_heads_, \
                "GQA头数不能大于注意力头数"
        
        # 设置设备和数据类型
        if self.device_map_ is None:
            self.device_map_ = "auto"


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


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.dim = config.head_dim_
        self.max_position_embeddings = config.max_seq_len_
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算sin/cos值
        t = position_ids.float().unsqueeze(-1) @ self.inv_freq.unsqueeze(0)
        freqs = torch.cat((t, t), dim=-1)
        
        # 应用旋转
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
            
        cos = freqs.cos()
        sin = freqs.sin()
        
        query_rot = rotate_half(query)
        key_rot = rotate_half(key)
        
        query = query * cos + query_rot * sin
        key = key * cos + key_rot * sin
        
        return query, key


class LLaVAEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)


class LLaVARMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight_ = nn.Parameter(torch.ones(hidden_size))
        self.norm_eps_ = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.norm_eps_)
        return (self.weight_ * hidden_states).to(input_dtype)


class GemmaAttention(nn.Module):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.hidden_size_ = config.hidden_size_
        self.num_heads_ = config.num_attention_heads_
        self.num_key_value_heads_ = config.num_key_value_heads_
        self.head_dim_ = config.head_dim_
        self.num_key_value_groups_ = self.num_heads_ // self.num_key_value_heads_
        
        # 投影矩阵
        self.q_proj_ = nn.Linear(self.hidden_size_, self.num_heads_ * self.head_dim_, bias=False)
        self.k_proj_ = nn.Linear(self.hidden_size_, self.num_key_value_heads_ * self.head_dim_, bias=False)
        self.v_proj_ = nn.Linear(self.hidden_size_, self.num_key_value_heads_ * self.head_dim_, bias=False)
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
        
        # 重塑形状
        query = query.view(batch_size, seq_length, self.num_heads_, self.head_dim_)
        key = key.view(batch_size, seq_length, self.num_key_value_heads_, self.head_dim_)
        value = value.view(batch_size, seq_length, self.num_key_value_heads_, self.head_dim_)
        
        # 重复键值以匹配查询头数
        key = key.repeat_interleave(self.num_key_value_groups_, dim=2)
        value = value.repeat_interleave(self.num_key_value_groups_, dim=2)
        
        # 转置以进行注意力计算
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 应用RoPE
        if self.rotary_emb_ is not None and position_ids is not None:
            query, key = self.rotary_emb_(query, key, position_ids)
            
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim_)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value)
        
        # 重塑并投影输出
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_length, -1)
        hidden_states = self.o_proj_(hidden_states)
        
        return hidden_states


class LLaVAMLP(nn.Module):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.gate_proj_ = nn.Linear(config.hidden_size_, config.intermediate_size_, bias=False)
        self.up_proj_ = nn.Linear(config.hidden_size_, config.intermediate_size_, bias=False)
        self.down_proj_ = nn.Linear(config.intermediate_size_, config.hidden_size_, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj_(x))
        up = self.up_proj_(x)
        return self.down_proj_(gate * up)


class LLaVADecoderLayer(nn.Module):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.self_attn_ = GemmaAttention(config)
        self.mlp_ = LLaVAMLP(config)
        
        self.input_norm_ = LLaVARMSNorm(config.hidden_size_, config.rms_norm_eps_)
        self.post_attn_norm_ = LLaVARMSNorm(config.hidden_size_, config.rms_norm_eps_)
        
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


class CLIPVisionTower(nn.Module):
    """CLIP视觉编码器塔"""
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config
        
        # 加载CLIP视觉模型
        self.vision_model = CLIPVisionModel.from_pretrained(
            config.vision_tower_,
            torch_dtype=config.dtype_,
            device_map=config.device_
        )
        self.vision_model.requires_grad_(False)
        
        # 加载图像处理器
        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.vision_tower_
        )
        
        # 获取视觉特征维度
        self.hidden_size = self.vision_model.config.hidden_size
        
    def forward(
        self,
        images: torch.FloatTensor,
        output_hidden_states: bool = True
    ) -> Dict[str, torch.FloatTensor]:
        """前向传播
        Args:
            images: 图像张量，shape (batch_size, num_channels, height, width)
            output_hidden_states: 是否输出中间隐藏状态
        Returns:
            包含以下键的字典：
            - hidden_states: 所有层的隐藏状态
            - image_embeds: 最后一层的特征
        """
        outputs = self.vision_model(
            images,
            output_hidden_states=output_hidden_states
        )
        
        # 根据配置选择特定层的特征
        if output_hidden_states:
            hidden_states = outputs.hidden_states
            selected_hidden_state = hidden_states[self.config.mm_vision_select_layer_]
        else:
            selected_hidden_state = outputs.last_hidden_state
            
        image_embeds = selected_hidden_state
        
        # 根据配置选择特征类型
        if self.config.mm_vision_select_feature_ == "patch":
            image_embeds = image_embeds[:, 1:] # 移除 [CLS] token
        elif self.config.mm_vision_select_feature_ == "cls_patch":
            pass  # 保持原样，包含 [CLS] token
        else:
            raise ValueError(f"不支持的特征类型: {self.config.mm_vision_select_feature_}")
            
        outputs = {
            "hidden_states": hidden_states if output_hidden_states else None,
            "image_embeds": image_embeds
        }
        return outputs
    
    def preprocess_images(self, images: Union[torch.FloatTensor, List[str], List["PIL.Image.Image"]]) -> torch.FloatTensor:
        """预处理图像
        Args:
            images: 图像列表或张量
        Returns:
            预处理后的图像张量
        """
        if isinstance(images, torch.Tensor):
            return images
            
        # 使用CLIP图像处理器处理图像
        image_inputs = self.image_processor(
            images,
            return_tensors="pt",
            do_resize=True,
            do_center_crop=True,
            size={"height": 224, "width": 224}
        )
        return image_inputs.pixel_values.to(
            device=self.vision_model.device,
            dtype=self.vision_model.dtype
        )


class GemmaMultiModalProjector(nn.Module):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config
        
        # 视觉特征到语言空间的投影
        self.linear_1 = nn.Linear(config.mm_hidden_size_, config.hidden_size_, bias=True)
        self.layer_norm = nn.LayerNorm(config.hidden_size_)
        self.linear_2 = nn.Linear(config.hidden_size_, config.hidden_size_, bias=True)
        
        if not config.tune_mm_mlp_adapter_:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(vision_features)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.gelu(hidden_states)  # Gemma使用GELU
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LLaVAForCausalLM(VLMForCausalLM):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config_ = config
        
        # 初始化分词器
        self.tokenizer = LlamaTokenizer.from_pretrained(
            config.name_or_path_,
            trust_remote_code=True,
        )
        
        # 加载预训练的语言模型
        pretrained_model = LlamaForCausalLM.from_pretrained(
            config.name_or_path_,
            torch_dtype=config.dtype_,
            device_map=config.device_map_,
            trust_remote_code=True
        )
        
        # 复制预训练模型的参数
        self.model = pretrained_model
        
        # 初始化视觉编码器
        self.vision_tower = CLIPVisionTower(config)
        
        # 初始化视觉特征投影层
        self.mm_projector = GemmaMultiModalProjector(config)
        
        # 特殊token的处理
        if config.tune_mm_mlp_adapter_:
            self.mm_projector.requires_grad_(True)
        else:
            self.mm_projector.requires_grad_(False)
        
        # 设置device
        self.config = config
        if hasattr(config, "device_") and config.device_ is not None:
            self.to(config.device_)
        
        # 更新配置
        self.vocab_size_ = pretrained_model.config.vocab_size
        self.hidden_size_ = pretrained_model.config.hidden_size
        self.num_attention_heads_ = pretrained_model.config.num_attention_heads
        self.num_hidden_layers_ = pretrained_model.config.num_hidden_layers
        self.pad_token_id_ = getattr(pretrained_model.config, "pad_token_id", 0)
        
        # 初始化本地模型组件
        self.embed_tokens_ = LLaVAEmbedding(
            pretrained_model.model.embed_tokens.weight,
            self.pad_token_id_
        )
        self.layers_ = nn.ModuleList([
            LLaVADecoderLayer(config) for _ in range(self.num_hidden_layers_)
        ])
        self.norm_ = LLaVARMSNorm(
            self.hidden_size_,
            config.rms_norm_eps_
        )
        self.rotary_emb_ = GemmaRotaryEmbedding(config)
        self.lm_head_ = nn.Linear(self.hidden_size_, self.vocab_size_, bias=False)
        
        # 图像标记
        self.image_newline_token_ = "</image>"
        self.image_start_token_ = "<image>"
        self.image_end_token_ = "</image>"
        
        # 确保特殊token在词表中
        special_tokens = {
            "additional_special_tokens": [
                self.image_start_token_,
                self.image_end_token_,
                self.image_newline_token_
            ]
        }
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        
        # 加载预训练权重
        self._load_pretrained_weights(pretrained_model)
        
        # 调整词表大小
        self.resize_token_embeddings(self.vocab_size_ + num_new_tokens)
        
        if config.model_type_ == "gemma":
            self._init_gemma_components(config)
        else:
            self._init_llama_components(config)
        
    def _load_pretrained_weights(self, pretrained_model: nn.Module):
        """从预训练模型加载权重"""
        # 加载词嵌入
        self.embed_tokens_.token_embedding_ = pretrained_model.model.embed_tokens.weight.data.clone()
        
        # 加载解码器层参数
        for i, layer in enumerate(self.layers_):
            pretrained_layer = pretrained_model.model.layers[i]
            
            # 加载自注意力层参数
            layer.self_attn_.q_proj_.weight.data = pretrained_layer.self_attn.q_proj.weight.data.clone()
            layer.self_attn_.k_proj_.weight.data = pretrained_layer.self_attn.k_proj.weight.data.clone()
            layer.self_attn_.v_proj_.weight.data = pretrained_layer.self_attn.v_proj.weight.data.clone()
            layer.self_attn_.o_proj_.weight.data = pretrained_layer.self_attn.o_proj.weight.data.clone()
            
            # 加载MLP层参数
            layer.mlp_.gate_proj_.weight.data = pretrained_layer.mlp.gate_proj.weight.data.clone()
            layer.mlp_.up_proj_.weight.data = pretrained_layer.mlp.up_proj.weight.data.clone()
            layer.mlp_.down_proj_.weight.data = pretrained_layer.mlp.down_proj.weight.data.clone()
            
            # 加载归一化层参数
            layer.input_norm_.weight.data = pretrained_layer.input_layernorm.weight.data.clone()
            layer.post_attn_norm_.weight.data = pretrained_layer.post_attention_layernorm.weight.data.clone()
        
        # 加载最终归一化层参数
        self.norm_.weight.data = pretrained_model.model.norm.weight.data.clone()
        
        # 加载语言模型头参数
        self.lm_head_.weight.data = pretrained_model.lm_head.weight.data.clone()
        
        # 将模型设置为评估模式
        self.eval()
        
    def resize_token_embeddings(self, new_num_tokens: int):
        """调整词嵌入大小以适应新的词表大小"""
        # 调整语言模型词嵌入
        self.model.resize_token_embeddings(new_num_tokens)
        
        # 更新模型词嵌入
        self.embed_tokens_ = LLaVAEmbedding(
            self.model.model.embed_tokens.weight,
            self.pad_token_id_
        )
        
        # 更新语言模型头
        old_lm_head = self.lm_head_
        self.lm_head_ = nn.Linear(self.hidden_size_, new_num_tokens, bias=False)
        
        # 复制旧权重
        if old_lm_head is not None:
            self.lm_head_.weight.data[:old_lm_head.weight.shape[0]] = old_lm_head.weight.data
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[Union[torch.FloatTensor, List[str], List["PIL.Image.Image"]]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = input_ids.shape[0]
        
        # 处理图像输入
        if images is not None:
            # 预处理图像
            image_tensor = self.vision_tower.preprocess_images(images)
            
            # 获取图像特征
            image_outputs = self.vision_tower(
                image_tensor,
                output_hidden_states=True
            )
            image_embeds = image_outputs["image_embeds"]
            
            # 投影图像特征到语言空间
            image_embeds = self.mm_projector(image_embeds)
            
            # 准备图像特征序列
            image_start_tokens = torch.where(input_ids == self.tokenizer.convert_tokens_to_ids(self.image_start_token_))[1]
            image_end_tokens = torch.where(input_ids == self.tokenizer.convert_tokens_to_ids(self.image_end_token_))[1]
            
            # 获取词嵌入
            inputs_embeds = self.embed_tokens_(input_ids)
            
            # 将图像特征插入序列
            for idx, (start_idx, end_idx) in enumerate(zip(image_start_tokens, image_end_tokens)):
                # 计算图像token的数量
                num_image_tokens = end_idx - start_idx - 1
                
                # 如果需要，调整图像特征的大小以匹配token数量
                if num_image_tokens > 0:
                    # 在时间维度上平均池化以匹配所需的token数量
                    reshaped_image_embeds = F.adaptive_avg_pool1d(
                        image_embeds[idx].transpose(0, 1).unsqueeze(0),
                        num_image_tokens
                    ).squeeze(0).transpose(0, 1)
                    
                    # 替换对应位置的嵌入
                    inputs_embeds[idx, start_idx+1:end_idx] = reshaped_image_embeds
        else:
            inputs_embeds = self.embed_tokens_(input_ids)
        
        # 生成位置编码的位置 ID
        seq_length = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 通过所有解码器层
        hidden_states = inputs_embeds
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
        
        outputs = {"logits": logits}
        
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs["loss"] = loss
            
        return outputs

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入的 token ID 转换为词嵌入"""
        return self.embed_tokens_(input_ids)

    def decoder_stack(self) -> List[VLMDecoder]:
        """返回解码器层的列表"""
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用最终的层归一化"""
        return self.norm_(hidden_states)

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.LongTensor:
        """生成文本"""
        # 设置默认值
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id_
        eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        # 初始化生成序列
        generated_ids = input_ids.clone()
        batch_size = input_ids.shape[0]
        cur_length = input_ids.shape[1]
        
        # 生成循环
        while cur_length < max_length:
            # 获取模型输出
            outputs = self(generated_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 应用 top_p 采样
            if do_sample:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样下一个 token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加生成的 token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cur_length += 1
            
            # 检查是否生成了结束符
            if eos_token_id is not None and (generated_ids == eos_token_id).any(dim=1).all():
                break
                
        return generated_ids

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        **kwargs
    ) -> "LLaVAForCausalLM":
        """从预训练模型加载参数并初始化模型
        Args:
            pretrained_model_name_or_path: HuggingFace 模型名称或路径
            device: 设备
            dtype: 数据类型
        Returns:
            初始化好的模型
        """
        print(f"正在从 {pretrained_model_name_or_path} 加载模型...")
        
        # 创建配置
        config = LLaVAConfig(
            name_or_path_=pretrained_model_name_or_path,
            device_=device,
            dtype_=dtype,
            # 基础配置
            hidden_size_=2048,
            num_attention_heads_=8,
            num_hidden_layers_=18,
            max_seq_len_=8192,
            vocab_size_=256000,
            # 模型特定配置
            intermediate_size_=11008,
            rope_theta_=10000,
            rms_norm_eps_=1e-6,
            # 视觉相关配置
            vision_tower_="openai/clip-vit-large-patch14",
            mm_vision_select_layer_=-2,
            mm_vision_select_feature_="patch",
            image_aspect_ratio_="square",
            tune_mm_mlp_adapter_=False,
            mm_use_im_start_end_=False,
            mm_hidden_size_=1024,
        )
        
        # 创建模型实例
        model = cls(config)
        
        return model


class GemmaLLaVA(nn.Module):
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config
        
        # 加载Gemma基础模型
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.name_or_path_,
            device_map=config.device_map_,
            torch_dtype=config.dtype_,
            trust_remote_code=True
        )
        
        # 加载视觉编码器
        self.vision_tower = CLIPVisionModel.from_pretrained(
            config.vision_tower_,
            torch_dtype=config.dtype_,
            device_map=config.device_map_
        )
        self.vision_tower.requires_grad_(False)
        
        # 多模态投影
        self.mm_projector = GemmaMultiModalProjector(config)
        
        # 特殊token处理
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path_,
            trust_remote_code=True
        )
        
        # 添加特殊token
        special_tokens = {
            "additional_special_tokens": [
                "<image>",
                "</image>",
                "<image_newline>"
            ]
        }
        num_new_tokens = self.tokenizer.add_special_tokens(special_tokens)
        
        # 调整embedding大小
        if num_new_tokens > 0:
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        batch_size = input_ids.shape[0]
        
        # 处理图像输入
        if images is not None:
            vision_features = self.vision_tower(images).last_hidden_state
            vision_features = self.mm_projector(vision_features)
            
            # 将视觉特征插入到序列中
            # 具体实现取决于你的token设计
            
        # 前向传播
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

class LlamaLLaVA(LLaVAForCausalLM):
    """Llama特定的LLaVA实现"""
    pass
