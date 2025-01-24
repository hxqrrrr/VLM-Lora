import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer, GemmaForCausalLM
from vlm_lora.models.gemmallava import GemmaLavaForCausalLM
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def load_model(model_path: str, device: str = "cuda"):
    """加载预训练模型"""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Converting to LLaVA-Gemma")
    model = GemmaLavaForCausalLM.from_pretrained(model_path, device=device)
    model.eval()
    
    return model, tokenizer

def preprocess_image(image_path: str, device: str = "cuda"):
    """预处理图像"""
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                      default="C:/Users/hxq11/.cache/huggingface/hub/models--Intel--llava-gemma-2b/snapshots/d12774000ffa2ce06528a49f280fffd586282dd4",
                      help="Gemma模型路径")
    parser.add_argument("--image-path", type=str,
                      default="C:/Users/hxq11/Desktop/auto-ques/auto_word/wordcloud.png",
                      help="输入图像路径")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片中的内容",
                      help="提示词")
    parser.add_argument("--device", type=str, default="cuda",
                      help="设备类型: cuda 或 cpu")
    parser.add_argument("--max-length", type=int, default=512,
                      help="最大生成长度")
    args = parser.parse_args()
    
    # 1. 加载模型和分词器
    model, tokenizer = load_model(args.model_path, args.device)
    
    # 2. 处理图像
    image_tensor = preprocess_image(args.image_path, args.device)
    image_features = model.process_image(image_tensor)
    
    # 3. 准备输入
    messages = [
        {"role": "user", "content": f"<image>\n{args.prompt}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    ).to(args.device)
    
    # 4. 生成回复
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            image_features=image_features,
            max_length=args.max_length,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
    
    response = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    print("\nResponse:", response)

if __name__ == "__main__":
    main() 