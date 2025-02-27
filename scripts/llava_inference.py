import os
import argparse
import torch
import requests
from PIL import Image
from io import BytesIO
from typing import Optional, List
from vlm_lora.models.llava import LLaVA

# 示例图片URL
EXAMPLE_IMAGE_URLS = {
    "猫": "https://t7.baidu.com/it/u=1595072465,3644073269&fm=193&f=GIF",
    "狗": "https://t7.baidu.com/it/u=1951548898,3927145&fm=193&f=GIF",
    "风景": "https://t7.baidu.com/it/u=2582370511,530426427&fm=193&f=GIF",
    "美食": "https://t7.baidu.com/it/u=3601447414,1764260638&fm=193&f=GIF"
}

def parse_args():
    parser = argparse.ArgumentParser(description='LLaVA模型推理脚本')
    parser.add_argument('--model-path', type=str, default="/root/hxq/models/llava-weights",
                      help='LLaVA模型路径')
    parser.add_argument('--vision-path', type=str, default="/root/hxq/models/clip-weights",
                      help='CLIP视觉模型路径')
    parser.add_argument('--image-path', type=str, default=None,
                      help='图片路径或URL（可选）')
    parser.add_argument('--prompt', type=str, required=True,
                      help='输入提示文本')
    parser.add_argument('--max-tokens', type=int, default=100,
                      help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='生成的随机性 (0.0-1.0)')
    parser.add_argument('--top-p', type=float, default=0.9,
                      help='采样的概率阈值 (0.0-1.0)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='设备类型 (cuda/cpu)')
    return parser.parse_args()

def load_image(image_path: str) -> Optional[Image.Image]:
    """加载并验证图片（支持本地路径和URL）"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # 下载网络图片
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            # 加载本地图片
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            image = Image.open(image_path)
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"加载图片时出错: {str(e)}")
        return None

def prepare_prompt(prompt: str, has_image: bool = False) -> str:
    """准备输入提示"""
    if has_image:
        if "<image>" not in prompt:
            prompt = f"<image>{prompt}</image>"
    return prompt

def main():
    args = parse_args()
    
    try:
        # 验证模型路径
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"LLaVA模型路径不存在: {args.model_path}")
        if not os.path.exists(args.vision_path):
            raise FileNotFoundError(f"CLIP视觉模型路径不存在: {args.vision_path}")
            
        # 初始化模型
        print(f"正在加载模型...")
        model = LLaVA.from_pretrained(
            args.model_path,
            vision_tower_=args.vision_path,
            device_map_=args.device,
            dtype_=torch.float16 if args.device == 'cuda' else torch.float32
        )
        
        # 处理图片
        image = None
        if args.image_path:
            image = load_image(args.image_path)
            if image is None:
                return
                
        # 准备输入
        prompt = prepare_prompt(args.prompt, image is not None)
        inputs = model.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        # 生成回复
        print("正在生成回复...")
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                images=[image] if image else None,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
        
        # 解码并打印回复
        response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n生成的回复:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"运行时出错: {str(e)}")

if __name__ == "__main__":
    # 打印示例用法
    print("示例图片URL:")
    for category, url in EXAMPLE_IMAGE_URLS.items():
        print(f"{category}: {url}")
    print("\n使用示例:")
    print('python llava_inference.py --image-path "https://t7.baidu.com/it/u=1595072465,3644073269&fm=193&f=GIF" --prompt "这是什么动物？它在做什么？"')
    print("-" * 50)
    
    main()