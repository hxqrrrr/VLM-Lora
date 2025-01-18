import os
import random
from typing import Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DTDDataset(Dataset):
    """DTD (Describable Textures Dataset) 数据集加载器"""
    
    def __init__(self, 
                 root_dir: str, 
                 split: str = "train",
                 image_size: int = 224,
                 image_mean: List[float] = None,
                 image_std: List[float] = None,
                 max_length: int = 77):
        """
        初始化DTD数据集
        Args:
            root_dir: 数据集根目录
            split: 数据集分割，可选 "train", "val", "test"
            image_size: 输入图像大小
            image_mean: 图像均值，用于标准化
            image_std: 图像标准差，用于标准化
            max_length: 文本最大长度
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.max_length = max_length
        
        # 设置图像预处理
        if image_mean is None:
            image_mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP默认值
        if image_std is None:
            image_std = [0.26862954, 0.26130258, 0.27577711]  # CLIP默认值
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])
        
        # 验证目录结构
        self._verify_directory_structure()
        
        # 加载图像列表
        split_file = os.path.join(root_dir, "labels", f"{split}1.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, "r") as f:
            self.image_files = []
            for line in f:
                img_path = line.strip()
                full_path = os.path.join(root_dir, "images", img_path)
                if os.path.exists(full_path):
                    self.image_files.append(img_path)
                else:
                    print(f"Warning: Image file not found: {full_path}")
            
        if not self.image_files:
            raise RuntimeError(f"No valid images found in {split} split")
            
        # 加载类别映射
        classes_file = os.path.join(root_dir, "labels", "classes.txt")
        if not os.path.exists(classes_file):
            # 如果classes.txt不存在，从目录结构推断类别
            self.categories = sorted(list(set(
                path.split("/")[0] for path in self.image_files
            )))
            print(f"Warning: classes.txt not found, inferred {len(self.categories)} categories from directory structure")
        else:
            with open(classes_file, "r") as f:
                self.categories = [line.strip() for line in f.readlines()]
                
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # 验证所有图像的类别是否在类别列表中
        for img_path in self.image_files:
            category = img_path.split("/")[0]
            if category not in self.cat2idx:
                raise ValueError(f"Unknown category {category} in {img_path}")
        
        # 生成描述模板
        self.templates = [
            "This is a {} texture.",
            "The image shows a {} pattern.",
            "This texture can be described as {}.",
            "The surface appears to be {}.",
            "This is an example of {} texture."
        ]
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
        print(f"Found {len(self.categories)} categories: {', '.join(self.categories)}")
        
    def _verify_directory_structure(self):
        """验证数据集目录结构"""
        if not os.path.isdir(self.root_dir):
            raise NotADirectoryError(f"Dataset root directory not found: {self.root_dir}")
            
        images_dir = os.path.join(self.root_dir, "images")
        if not os.path.isdir(images_dir):
            raise NotADirectoryError(f"Images directory not found: {images_dir}")
            
        labels_dir = os.path.join(self.root_dir, "labels")
        if not os.path.isdir(labels_dir):
            raise NotADirectoryError(f"Labels directory not found: {labels_dir}")
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项
        Returns:
            包含以下键的字典：
            - image: 预处理后的图像张量
            - text: 文本描述
            - category_idx: 类别索引
        """
        # 获取图像文件路径
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, "images", img_name)
        
        try:
            # 加载并预处理图像
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个占位图像
            image_tensor = torch.zeros((3, self.image_size, self.image_size))
        
        # 获取类别
        category = img_name.split("/")[0]
        category_idx = self.cat2idx[category]
        
        # 随机选择一个模板生成描述
        template = random.choice(self.templates)
        description = template.format(category)
        
        return {
            "image": image_tensor,
            "text": description,
            "category_idx": category_idx
        }
        
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "text": [item["text"] for item in batch],
            "category_idx": torch.tensor([item["category_idx"] for item in batch])
        }
