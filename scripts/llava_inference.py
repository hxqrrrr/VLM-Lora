import os
import torch
from PIL import Image
import gc
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from transformers import AutoTokenizer
from vlm_lora.models.llava import LLaVAForCausalLM

def main():
    # 配置模型参数 - 使用较小的模型
    model_name = "liuhaotian/llava-v1.5-4b"  # 使用 4B 参数的模型
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # 设置 CUDA 内存分配器
    if device == "cuda":
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 设置内存分配器
        torch.cuda.set_per_process_memory_fraction(0.7)  # 降低到 70%
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        
        # 设置环境变量以避免内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print(f"正在从 {model_name} 加载模型...")

    try:
        # 加载模型和分词器
        model = LLaVAForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto",  # 使用自动设备映射
            torch_dtype=torch.float16,  # 使用半精度
            max_memory={0: "4GB", "cpu": "8GB"},  # 限制每个设备的最大内存
            offload_folder="offload",  # 设置模型权重卸载目录
        )
        
        print("模型加载成功!")
        
    except Exception as e:
        print(f"加载失败: {str(e)}")
        print("请检查模型文件是否完整")
        return
        
    try:
        model.eval()
        
        # 准备测试图片
        image_path = r"C:\Users\hxq11\Desktop\auto-ques\auto_word\wordcloud.png"
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到测试图片: {image_path}")
        
        image = Image.open(image_path)
        
        # 准备提示文本
        prompt = f"{model.image_start_token_}请用中文详细描述这张图片的内容。{model.image_end_token_}"
        
        # 处理输入
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        print("开始生成...")
        with torch.no_grad():
            # 将输入移到正确的设备上
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # 生成文本
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=[image],
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        # 解码生成的文本
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n生成的描述:")
        print(generated_text)
        
    except Exception as e:
        print(f"生成过程中出错: {str(e)}")
    finally:
        # 清理内存
        if device == "cuda":
            del model
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 