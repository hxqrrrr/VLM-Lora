from vlm_lora.common.load_json import load_json_config, print_config

def main():
    # 加载配置
    model_config, lora_config = load_json_config("templates/tinyllama_lora.json")
    
    # 打印配置
    print_config(model_config, lora_config)

if __name__ == "__main__":
    main() 