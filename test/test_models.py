import os
import sys
import json
# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from vlm_lora.common import (
    VLMModelConfig,
    VLMModelInput,
    AdapterConfig,
    LoraConfig,
    InputData,
    Prompt,
    BatchConfig,
)
from vlm_lora.models.tinyllama import TinyLLaMAForCausalLM
from vlm_lora.models import VLMModel
from vlm_lora.adapters import lora_config_factory

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_model_basic():
    # 加载配置文件
    config_data = load_config(os.path.join(project_root, 'templates', 'tinyllama_lora.json'))
    
    # 处理 dtype_
    model_config_dict = config_data['model_config']
    if "dtype_" in model_config_dict:
        dtype_str = model_config_dict["dtype_"]
        if dtype_str == "float32":
            model_config_dict["dtype_"] = torch.float32
        elif dtype_str == "float16":
            model_config_dict["dtype_"] = torch.float16
        elif dtype_str == "bfloat16":
            model_config_dict["dtype_"] = torch.bfloat16
    
    # 创建模型配置
    model_config = VLMModelConfig(**model_config_dict)

    # 创建模型实例
    base_model = TinyLLaMAForCausalLM(model_config)
    model = VLMModel(base_model)
    print("模型创建成功")

    # 创建一个简单的输入
    prompt = Prompt(
        instruction="翻译下面的句子到中文",
        input="Hello, world!",
    )
    input_data = InputData(inputs=[prompt])

    # 创建批次配置
    batch_config = BatchConfig(
        adapter_name_="",  # 默认适配器名称
        batch_start_idx_=0,  # 批次开始索引
        batch_end_idx_=1,  # 批次结束索引
    )

    # 对输入进行编码
    tokens = base_model.tokenizer(
        prompt.format_prompt(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # 创建模型输入
    model_input = VLMModelInput(
        batch_configs_=[batch_config],
        batch_tokens_=tokens["input_ids"],
        batch_labels_=None,
        batch_masks_=tokens["attention_mask"],
        inference_mode_=True
    )

    # 测试前向传播
    try:
        outputs = model(model_input)
        print("前向传播成功")
        if isinstance(outputs, (list, tuple)):
            print(f"输出形状: {outputs[0].logits.shape if outputs[0].logits is not None else None}")
        else:
            print(f"输出形状: {outputs.logits.shape if outputs.logits is not None else None}")
    except Exception as e:
        print(f"前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()

    # 测试 LoRA 配置
    lora_config = lora_config_factory(config_data['lora_config'])

    try:
        model.add_adapter(lora_config)
        print("LoRA 适配器添加成功")
    except Exception as e:
        print(f"LoRA 适配器添加失败: {str(e)}")

if __name__ == "__main__":
    test_model_basic() 