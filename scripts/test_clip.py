import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

from vlm_lora.models.clip import CLIPModel


def load_image(url: str):
    """从URL加载图像"""
    return Image.open(requests.get(url, stream=True).raw)


def main():
    # 可选的模型名称：
    # "openai/clip-vit-base-patch32"  # ViT-B/32
    # "openai/clip-vit-base-patch16"  # ViT-B/16
    # "openai/clip-vit-large-patch14"  # ViT-L/14
    # "openai/clip-vit-large-patch14-336"  # ViT-L/14@336px
    
    print("加载原始 CLIP 模型...")
    model_name = "openai/clip-vit-base-patch32"  # 可以根据需要更改模型大小
    cache_dir = r"C:\Users\hxq11\.cache\huggingface\hub"
    
    print(f"使用模型: {model_name}")
    hf_model = HFCLIPModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False
    )
    processor = CLIPProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False
    )
    
    # 打印模型配置信息
    print("\n模型配置:")
    print(f"视觉编码器:")
    print(f"- 嵌入维度: {hf_model.config.vision_config.hidden_size}")
    print(f"- Patch大小: {hf_model.config.vision_config.patch_size}")
    print(f"- 层数: {hf_model.config.vision_config.num_hidden_layers}")
    print(f"- 注意力头数: {hf_model.config.vision_config.num_attention_heads}")
    print(f"- 输入图像大小: {hf_model.config.vision_config.image_size}")
    
    print(f"\n文本编码器:")
    print(f"- 嵌入维度: {hf_model.config.text_config.hidden_size}")
    print(f"- 层数: {hf_model.config.text_config.num_hidden_layers}")
    print(f"- 注意力头数: {hf_model.config.text_config.num_attention_heads}")
    print(f"- 最大序列长度: {hf_model.config.text_config.max_position_embeddings}")
    
    # 创建我们的模型
    print("\n创建 VLM-LoRA CLIP 模型...")
    model = CLIPModel.from_pretrained(hf_model)
    model.eval()
    
    # 准备输入数据
    print("\n准备测试数据...")
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000039769.jpg",
    ]
    texts = ["一只猫", "一只狗", "一辆汽车"]
    
    # 加载图片并调整大小以匹配模型要求
    images = [load_image(url) for url in image_urls]
    
    # 处理输入
    inputs = processor(
        text=texts,
        images=images,
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
        
        # 打印调试信息
        print(f"\n调试信息:")
        print(f"输入的形状:")
        print(f"- input_ids: {inputs.input_ids.shape}")
        print(f"- pixel_values: {inputs.pixel_values.shape}")
        print(f"- attention_mask: {inputs.attention_mask.shape}")
        print(f"输出的形状:")
        print(f"- logits_per_image: {outputs['logits_per_image'].shape}")
        print(f"- logits_per_text: {outputs['logits_per_text'].shape}")
    
    # 验证输出
    print("\n验证输出:")
    print(f"损失值: {outputs['loss'].item():.4f}")
    print(f"图像-文本相似度矩阵形状: {outputs['logits_per_image'].shape}")
    
    # 计算并显示相似度
    probs = outputs["logits_per_image"].softmax(dim=1)
    print("\n图像-文本相似度:")
    for i, (text, prob) in enumerate(zip(texts, probs)):
        print(f"\n图片 {i+1} 的相似度:")
        for j, (t, p) in enumerate(zip(texts, prob)):
            print(f"{t}: {p.item():.4%}")


if __name__ == "__main__":
    main() 