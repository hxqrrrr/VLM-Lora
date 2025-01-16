import torch
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from vlm_lora.common import (
    VLMModelConfig,
    VLMCache,
    VLMDecoder,
    VLMForCausalLM,
)
from vlm_lora.models.base import VLMModel

class TinyLLaMAForCausalLM(VLMForCausalLM):
    def __init__(self, config: VLMModelConfig):
        super().__init__()
        self.config_ = config
        
        # 确保模型文件已下载
        try:
            print(f"尝试直接加载模型: {config.name_or_path_}")
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
        except Exception as e:
            print(f"直接加载失败: {str(e)}")
            print(f"尝试下载模型: {config.name_or_path_}")
            try:
                cache_dir = snapshot_download(
                    repo_id=config.name_or_path_,
                    token=True
                )
                print(f"模型下载成功，缓存位置: {cache_dir}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    cache_dir,
                    device_map=config.device_,
                    torch_dtype=config.dtype_,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cache_dir,
                    trust_remote_code=True
                )
            except Exception as download_error:
                print(f"模型下载失败: {str(download_error)}")
                raise
        
        print("模型加载成功，开始配置模型参数")
        # 更新配置
        self.config_.vocab_size_ = self.model.config.vocab_size
        self.config_.dim_ = self.model.config.hidden_size
        self.config_.head_dim_ = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.config_.intermediate_ = getattr(self.model.config, "intermediate_size", 4 * self.model.config.hidden_size)
        self.config_.n_heads_ = self.model.config.num_attention_heads
        self.config_.n_kv_heads_ = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        self.config_.n_layers_ = self.model.config.num_hidden_layers
        self.config_.hidden_act_ = getattr(self.model.config, "hidden_act", "silu")
        self.config_.hidden_dropout_ = getattr(self.model.config, "dropout", 0.0)  # LLaMA 使用 dropout
        self.config_.pad_token_id_ = getattr(self.model.config, "pad_token_id", 0)
        self.config_.rope_theta_ = getattr(self.model.config, "rope_theta", 10000.0)
        self.config_.partial_rotary_factor_ = getattr(self.model.config, "partial_rotary_factor", 1.0)
        self.config_.max_seq_len_ = getattr(self.model.config, "max_position_embeddings", 2048)
        
        print("配置参数详情:")
        print(f"- vocab_size: {self.config_.vocab_size_}")
        print(f"- hidden_size: {self.config_.dim_}")
        print(f"- num_attention_heads: {self.config_.n_heads_}")
        print(f"- num_hidden_layers: {self.config_.n_layers_}")
        print(f"- max_seq_len: {self.config_.max_seq_len_}")
        
        # 保存一些常用的配置到实例变量
        self.vocab_size_ = self.config_.vocab_size_
        self.hidden_size_ = self.config_.dim_
        self.num_attention_heads_ = self.config_.n_heads_
        self.num_hidden_layers_ = self.config_.n_layers_
        self.pad_token_id_ = self.config_.pad_token_id_
        
        # 语言模型头
        self.lm_head_ = self.model.lm_head
        print("模型参数配置完成")

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将输入的 token ID 转换为词嵌入"""
        return self.model.model.embed_tokens(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TinyLLaMA 使用 RoPE 位置编码"""
        # 获取 RoPE 编码的参数
        head_dim = self.hidden_size_ // self.num_attention_heads_
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # 计算 sin/cos 值
        t = position_ids.float().unsqueeze(-1) @ inv_freq.to(position_ids.device).unsqueeze(0)
        sin = torch.sin(t).unsqueeze(-2)
        cos = torch.cos(t).unsqueeze(-2)
        
        # 将 sin/cos 移动到正确的设备
        sin = sin.to(device=input_tensor.device)
        cos = cos.to(device=input_tensor.device)
        
        return sin, cos

    def decoder_stack(self) -> List[VLMDecoder]:
        """返回解码器层的列表"""
        return self.model.model.layers

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用最终的层归一化"""
        return self.model.model.norm(hidden_states)

    def causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[VLMCache],
    ) -> torch.Tensor:
        """生成因果注意力掩码"""
        batch_size, seq_length = input_tensor.shape[:2]
        
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_tensor.device),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 如果提供了注意力掩码，则与因果掩码组合
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            causal_mask = causal_mask & attention_mask
            
        return causal_mask

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "TinyLLaMAForCausalLM":
        """从预训练模型创建实例"""
        config = VLMModelConfig(
            name_or_path_=model_path,
            device_=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            dtype_=kwargs.get("dtype", torch.float32)
        )
        return cls(config)
