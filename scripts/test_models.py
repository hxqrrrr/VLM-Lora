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

def test_model_basic():
    # 创建模型配置
    model_config = VLMModelConfig(
        name_or_path_="TinyLlama/TinyLlama-1.1B-Chat-v0.4",  # 使用最新的 v0.4 版本
        device_="cuda" if torch.cuda.is_available() else "cpu",
        dtype_=torch.float32
    )

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
        adapter_name_="test_adapter",
        batch_start_idx_=0,
        batch_end_idx_=1
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
        batch_configs_=[batch_config],  # 使用 BatchConfig
        batch_tokens_=tokens["input_ids"],
        batch_labels_=None,
        batch_masks_=tokens["attention_mask"],
        inference_mode_=True
    )

    # 测试前向传播
    try:
        output = model(model_input)
        print("前向传播成功")
        print(f"输出形状: {output[0].logits.shape if output[0].logits is not None else None}")
    except Exception as e:
        print(f"前向传播失败: {str(e)}")

    # 测试 LoRA 配置
    lora_config = LoraConfig(
        adapter_name="test_adapter",
        task_name="casual",
        lora_r_=8,
        lora_alpha_=32,
        lora_dropout_=0.1,
        target_modules_={"q_proj": True, "v_proj": True}
    )

    try:
        model.add_adapter(lora_config)
        print("LoRA 适配器添加成功")
    except Exception as e:
        print(f"LoRA 适配器添加失败: {str(e)}")

if __name__ == "__main__":
    test_model_basic() 