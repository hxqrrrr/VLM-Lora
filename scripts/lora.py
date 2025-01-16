import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vlm_lora.common.config import , ModelInput, BatchConfig
from vlm_lora.common.lora import Linear
import types

def load_config(config_path="templates/tinyllama_lora.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_model(config):
    # 检查 CUDA 是否可用
    print("\n检查 CUDA 状态...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"   √ CUDA 可用！")
        print(f"   - GPU 数量: {device_count}")
        print(f"   - 当前使用: {device_name}")
        print(f"   - 显存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    else:
        print("   × CUDA 不可用，将使用 CPU 运行")
    
    # 1. 加载 TinyLlama 模型和分词器
    print("\n1. 开始加载 TinyLlama 模型和分词器...")
    model_name = config["model_name"]
    
    print("   - 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("   - 加载基础模型（这可能需要几分钟）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config["model_config"]["torch_dtype"]),
        device_map=config["model_config"]["device_map"]
    )
    print("   √ 基础模型加载完成！")
    if cuda_available:
        print(f"   - 当前显存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    # 2. 配置 LoRA
    print("\n2. 配置 LoRA 参数...")
    lora_config = .from_config(config["lora_config"])
    print("   √ LoRA 配置完成！")
    
    # 3. 应用 LoRA
    print("\n3. 正在应用 LoRA 到模型...")
    module_count = 0
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
            target in name for target, enabled in lora_config.target_modules_.items() if enabled
        ):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = base_model.get_submodule(parent_name)
            
            # 替换为 LoRA 模块
            new_module = Linear(
                base_layer=module,
                device=str(module.weight.device)
            )
            new_module.init_lora_weight(lora_config)
            setattr(parent_module, child_name, new_module)
            module_count += 1
    print(f"   √ 成功应用 LoRA 到 {module_count} 个模块！")
    
    # 4. 修改模型的前向传播方法
    print("\n4. 配置模型前向传播...")
    original_forward = base_model.forward
    def forward(self, *args, **kwargs):
        input_args = kwargs.get('input_args', None)
        if input_args is None:
            input_args = ModelInput(
                batch_configs_=[
                    BatchConfig(
                        adapter_name_=lora_config.adapter_name,
                        batch_start_idx_=0,
                        batch_end_idx_=1
                    )
                ],
                efficient_operator_=True
            )
        kwargs['input_args'] = input_args
        return original_forward(*args, **kwargs)
    
    base_model.forward = types.MethodType(forward, base_model)
    print("   √ 前向传播配置完成！")
    
    print("\n✓ 所有初始化步骤已完成！模型已准备就绪！")
    return base_model, tokenizer, config

def generate_text(model, tokenizer, prompt, config):
    # 构建系统提示和用户输入
    chat_template = config["chat_template"]
    chat_prompt = chat_template["template"].format(
        system_prompt=chat_template["system_prompt"],
        user_input=prompt
    )
    
    # 准备输入
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    # 创建 input_args
    model_input = ModelInput(
        batch_configs_=[
            BatchConfig(
                adapter_name_=config["lora_config"]["name"],
                batch_start_idx_=0,
                batch_end_idx_=1
            )
        ],
        efficient_operator_=True
    )
    
    # 在生成时传入 input_args
    outputs = model.generate(
        **inputs,
        **config["generation_config"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        input_args=model_input
    )
    
    # 解码并提取助手回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split(chat_template["response_split"])[-1].strip()
    return response

def main():
    # 加载配置
    config = load_config()
    
    # 设置模型
    print("正在加载模型...")
    model, tokenizer, config = setup_model(config)
    
    # 交互式对话
    print("\n模型已准备就绪！输入 'quit' 退出对话。")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'quit':
            break
            
        print("\n助手: ", end="")
        response = generate_text(model, tokenizer, user_input, config)
        print(response)
    
    # 保存 LoRA 权重
    print("\n保存 LoRA 权重...")
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, Linear) and hasattr(module, "loras_"):
            if config["lora_config"]["name"] in module.loras_:
                lora_state_dict[name] = {
                    "lora_a": module.loras_[config["lora_config"]["name"]].lora_a_.state_dict(),
                    "lora_b": module.loras_[config["lora_config"]["name"]].lora_b_.state_dict()
                }
    torch.save(lora_state_dict, "tinyllama_lora_weights.pt")
    print("LoRA 权重已保存到 tinyllama_lora_weights.pt")

if __name__ == "__main__":
    main()