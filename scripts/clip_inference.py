import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel as HFCLIPModel, CLIPConfig as HFCLIPConfig, CLIPTokenizer
from vlm_lora.models.clip import CLIPModel, CLIPConfig
from vlm_lora.data.dataset_config import DatasetConfig
from vlm_lora.data.dataset_factory import DatasetFactory
from vlm_lora.common.config import VLMModelConfig
import json
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME

def main():
    # 加载配置
    config_json_path = "templates/clip.json"
    model_config = CLIPConfig.from_json(config_json_path)
    
    # 获取模型名称和设备
    model_name = model_config.name_or_path_
    device = model_config.device_
    
    # 检查本地缓存
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    
    print(f"检查缓存路径: {model_cache}")
    print(f"缓存路径是否存在: {os.path.exists(model_cache)}")
    if os.path.exists(model_cache):
        print(f"缓存目录内容:")
        for file in os.listdir(model_cache):
            print(f"  - {file}")
    
    print(f"从HuggingFace加载模型: {model_name}")
    hf_config = HFCLIPConfig.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    pretrained_model = HFCLIPModel.from_pretrained(model_name)
    
    # 创建本地模型配置
    model_config = CLIPConfig(
        vision_embed_dim_=hf_config.vision_config.hidden_size,
        vision_patch_size_=hf_config.vision_config.patch_size,
        vision_layers_=hf_config.vision_config.num_hidden_layers,
        vision_heads_=hf_config.vision_config.num_attention_heads,
        vision_width_=hf_config.vision_config.intermediate_size,
        vision_dropout_=hf_config.vision_config.dropout,
        image_size_=hf_config.vision_config.image_size,
        text_embed_dim_=hf_config.text_config.hidden_size,
        text_layers_=hf_config.text_config.num_hidden_layers,
        text_heads_=hf_config.text_config.num_attention_heads,
        text_width_=hf_config.text_config.intermediate_size,
        text_dropout_=hf_config.text_config.dropout,
        max_position_embeddings_=hf_config.text_config.max_position_embeddings,
        projection_dim_=hf_config.projection_dim,
        vocab_size_=hf_config.text_config.vocab_size,
        device_=device,
        name_or_path_=model_name
    )
    
    # 加载数据集配置
    dataset_config = DatasetConfig.from_json(config_json_path)
    
    # 创建数据集
    dataset = DatasetFactory.create_dataset(dataset_config, split="test")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    # 创建并加载模型
    model = CLIPModel(model_config)
    model.tokenizer_ = tokenizer
    model = CLIPModel.from_pretrained(pretrained_model, device=device)
    model.tokenizer_ = tokenizer
    model = model.to(device)
    
    # 进行推理
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # 将数据移到设备上
            images = batch["image"].to(device)
            texts = batch["text"]
            
            # 获取图像和文本特征
            image_features = model.get_image_features(images)
            text_features = model.get_text_features(texts)
            
            # 计算相似度
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 打印结果
            for i, text in enumerate(texts):
                values, indices = similarity[i].topk(3)
                print(f"\n图像 {i}:")
                print(f"实际描述: {text}")
                print("Top 3 匹配:")
                for value, idx in zip(values, indices):
                    print(f"{texts[idx]}: {value.item():.2f}%")

if __name__ == "__main__":
    main() 