import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

from vlm_lora.models.clip import CLIPModel


def load_image(url: str):
    """从URL加载图像"""
    return Image.open(requests.get(url, stream=True).raw)


def main():
    # 加载原始模型和处理器
    print("加载原始 CLIP 模型...")
    model_name = "openai/clip-vit-base-patch32"
    hf_model = HFCLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # 创建我们的模型
    print("创建 VLM-LoRA CLIP 模型...")
    model = CLIPModel.from_pretrained(hf_model)
    model.eval()
    
    # 准备输入数据
    print("\n准备测试数据...")
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(image_url)
    texts = ["一只猫", "一只狗", "一辆汽车"]
    
    # 处理输入
    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    
    # 测试前向传播
    print("执行前向传播...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            return_loss=True
        )
    
    # 验证输出
    print("\n验证输出:")
    print(f"损失值: {outputs['loss'].item():.4f}")
    print(f"图像-文本相似度矩阵形状: {outputs['logits_per_image'].shape}")
    
    # 计算并显示相似度
    probs = outputs["logits_per_image"].softmax(dim=1)
    print("\n图像-文本相似度:")
    for text, prob in zip(texts, probs[0]):
        print(f"{text}: {prob.item():.4%}")


if __name__ == "__main__":
    main() 