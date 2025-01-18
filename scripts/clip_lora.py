import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel as HFCLIPModel, CLIPConfig as HFCLIPConfig, CLIPTokenizer
from vlm_lora.models.clip import CLIPModel, CLIPConfig
from vlm_lora.data.dataset_config import DatasetConfig
from vlm_lora.data.dataset_factory import DatasetFactory
from vlm_lora.common.config import VLMModelConfig
from vlm_lora.common.lora import LoraConfig
import json
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME

def main():
    # 1. 加载配置
    config_json_path = "templates/clip.json"
    model_config = CLIPConfig.from_json(config_json_path)
    lora_config = LoraConfig.from_json(config_json_path)
    dataset_config = DatasetConfig.from_json(config_json_path)
    
    # 2. 准备模型加载
    model_name = model_config.name_or_path_
    device = model_config.device_
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"), "snapshots", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    
    # 3. 加载预训练模型和分词器
    print(f"从本地缓存加载模型: {model_cache}")
    if not os.path.exists(model_cache):
        print(f"本地缓存不存在，从HuggingFace下载: {model_name}")
        hf_config = HFCLIPConfig.from_pretrained(model_name)
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        pretrained_model = HFCLIPModel.from_pretrained(model_name)
    else:
        hf_config = HFCLIPConfig.from_pretrained(model_cache)
        tokenizer = CLIPTokenizer.from_pretrained(model_cache)
        pretrained_model = HFCLIPModel.from_pretrained(model_cache)
    
    # 4. 创建并初始化模型
    model = CLIPModel.from_pretrained(pretrained_model, device=device)
    model.tokenizer_ = tokenizer
    model = model.to(device)
    
    # 5. 设置LoRA
    print("\n=== LoRA配置 ===")
    print(f"LoRA配置: {json.dumps(lora_config.__dict__, indent=2)}")
    print("\n添加LoRA层...")
    model.add_lora_layers(lora_config)
    print("冻结基础模型参数...")
    model.freeze_base_model()
    
    # 检查LoRA层是否正确添加
    print("\n=== 模型参数检查 ===")
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    print(f"LoRA层数量: {len(lora_params)}")
    
    print("\n所有模型参数:")
    for name, param in model.named_parameters():
        print(f"参数: {name}, requires_grad: {param.requires_grad}, shape: {param.shape}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n可训练参数总数: {len(trainable_params)}")
    
    # 6. 准备数据集
    train_dataset = DatasetFactory.create_dataset(dataset_config, split="train")
    val_dataset = DatasetFactory.create_dataset(dataset_config, split="val")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
 
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数数量: {len(trainable_params)}")
    
    # 8. 训练准备
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=0.01
    )
    
    # 9. 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            images = batch["image"].to(device)
            texts = batch["text"]
            
            image_features = model.get_image_features(images)
            text_features = model.get_text_features(texts)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T * model.logit_scale_.exp()
            
            labels = torch.arange(len(images), device=device)
            loss = (
                torch.nn.functional.cross_entropy(logits, labels) +
                torch.nn.functional.cross_entropy(logits.T, labels)
            ) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["image"].to(device)
                texts = batch["text"]
                
                image_features = model.get_image_features(images)
                text_features = model.get_text_features(texts)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T * model.logit_scale_.exp()
                
                labels = torch.arange(len(images), device=device)
                loss = (
                    torch.nn.functional.cross_entropy(logits, labels) +
                    torch.nn.functional.cross_entropy(logits.T, labels)
                ) / 2
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
    
    # 10. 保存模型
    save_dir = "checkpoints/clip_lora"
    os.makedirs(save_dir, exist_ok=True)
    model.save_lora_weights(save_dir)
    print(f"LoRA权重已保存到: {save_dir}")

if __name__ == "__main__":
    main() 