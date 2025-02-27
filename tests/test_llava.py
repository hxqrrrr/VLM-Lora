import os
import torch
import pytest
from PIL import Image
from pathlib import Path
from transformers import AutoConfig, PretrainedConfig

from vlm_lora.models.llava import LLaVA, LLaVAConfig
from vlm_lora.common import VLMCache

class MockPretrainedConfig(PretrainedConfig):
    """模拟预训练配置"""
    model_type = "mock"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.num_hidden_layers = 2
        self.vocab_size = 1000

@pytest.fixture
def mock_model_path(tmp_path):
    """创建模拟模型路径"""
    model_path = tmp_path / "mock_model"
    model_path.mkdir()
    
    # 创建配置文件
    config = MockPretrainedConfig()
    config.save_pretrained(model_path)
    
    return str(model_path)

@pytest.fixture
def config(mock_model_path):
    """创建测试配置"""
    return LLaVAConfig(
        name_or_path_=mock_model_path,
        vision_tower_=mock_model_path,
        hidden_size_=32,
        num_attention_heads_=4,
        num_hidden_layers_=2,
        mm_hidden_size_=32,
        device_map_="cpu"
    )

@pytest.fixture
def model(config):
    return LLaVA(config)

def test_model_init(model):
    """测试模型初始化"""
    assert model is not None
    assert model.tokenizer is not None
    assert model.language_model is not None
    assert model.vision_tower is not None
    assert model.mm_projector is not None
    assert model.decoder is not None
    
    # 验证模型配置
    assert model.language_model.config.hidden_size == 32
    assert model.language_model.config.num_attention_heads == 4
    assert model.language_model.config.num_hidden_layers == 2

@pytest.mark.skip(reason="需要真实tokenizer")
def test_tokenizer(model):
    """测试分词器"""
    text = "Hello, this is a test."
    tokens = model.tokenizer(text, return_tensors="pt")
    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    
    # 测试特殊token
    special_tokens = ["<image>", "</image>", "<image_newline>"]
    for token in special_tokens:
        assert token in model.tokenizer.get_vocab()

@pytest.mark.skip(reason="需要真实模型")
def test_text_only_forward(model):
    """测试纯文本前向传播"""
    text = "Hello, this is a test."
    tokens = model.tokenizer(text, return_tensors="pt")
    
    outputs = model.forward(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"]
    )
    
    assert "logits" in outputs
    assert outputs.logits.shape[1] == tokens["input_ids"].shape[1]

@pytest.mark.skip(reason="需要真实模型")
def test_text_only_generate(model):
    """测试纯文本生成"""
    text = "Complete this sentence: The quick brown fox"
    tokens = model.tokenizer(text, return_tensors="pt")
    
    outputs = model.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9
    )
    
    assert outputs.shape[1] > tokens["input_ids"].shape[1]
    generated_text = model.tokenizer.decode(outputs[0])
    assert len(generated_text) > len(text)

def test_vision_tower(model):
    """测试视觉塔"""
    # 创建一个测试图像
    image = Image.new('RGB', (224, 224), color='red')
    image_tensor = model.vision_tower.preprocess([image])
    
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (1, 3, 224, 224)
    
    features = model.vision_tower(image_tensor)
    assert "image_features" in features
    assert features["image_features"].shape[0] == 1

def test_multimodal_forward(model):
    """测试多模态前向传播"""
    # 准备输入
    text = "<image>A test image</image> What do you see?"
    tokens = model.tokenizer(text, return_tensors="pt")
    image = Image.new('RGB', (224, 224), color='red')
    
    outputs = model.forward(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        images=[image]
    )
    
    assert "logits" in outputs
    assert outputs.logits.shape[1] == tokens["input_ids"].shape[1]

def test_multimodal_generate(model):
    """测试多模态生成"""
    # 准备输入
    text = "<image>A test image</image> Describe what you see in the image."
    tokens = model.tokenizer(text, return_tensors="pt")
    image = Image.new('RGB', (224, 224), color='red')
    
    outputs = model.generate(
        input_ids=tokens["input_ids"],
        images=[image],
        max_new_tokens=50
    )
    
    assert outputs.shape[1] > tokens["input_ids"].shape[1]
    generated_text = model.tokenizer.decode(outputs[0])
    assert len(generated_text) > len(text)

def test_cache_mechanism(model):
    """测试缓存机制"""
    text = "This is a test of the cache mechanism."
    tokens = model.tokenizer(text, return_tensors="pt")
    cache = VLMCache()
    
    # 第一次前向传播
    outputs1 = model.forward(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        cache=cache
    )
    
    assert cache.past_key_values is not None
    
    # 使用缓存的第二次前向传播
    next_token = outputs1.logits[:, -1:].argmax(dim=-1)
    outputs2 = model.forward(
        input_ids=next_token,
        cache=cache
    )
    
    assert outputs2.logits.shape[1] == 1

def test_from_pretrained(config):
    """测试从预训练模型加载"""
    model = LLaVA.from_pretrained(config.name_or_path_)
    assert model is not None
    assert isinstance(model, LLaVA)

def test_get_tokenizer_and_decoder(model):
    """测试获取tokenizer和decoder"""
    tokenizer = model.get_tokenizer()
    decoder = model.get_decoder()
    
    assert tokenizer is not None
    assert decoder is not None
    assert tokenizer is model.tokenizer
    assert decoder is model.decoder
