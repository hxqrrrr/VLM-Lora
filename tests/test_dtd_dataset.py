import os
import sys
import unittest
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm_lora.data.load_dataset import DTDDataset

class TestDTDDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.root_dir = "data/dtd"
        cls.batch_size = 4
        
    def test_dataset_loading(self):
        """测试数据集加载"""
        # 测试训练集
        train_dataset = DTDDataset(self.root_dir, split="train")
        self.assertGreater(len(train_dataset), 0, "训练集应该包含图像")
        
        # 测试验证集
        val_dataset = DTDDataset(self.root_dir, split="val")
        self.assertGreater(len(val_dataset), 0, "验证集应该包含图像")
        
        # 测试测试集
        test_dataset = DTDDataset(self.root_dir, split="test")
        self.assertGreater(len(test_dataset), 0, "测试集应该包含图像")
        
    def test_data_item(self):
        """测试数据项的获取"""
        dataset = DTDDataset(self.root_dir, split="train")
        image, description = dataset[0]
        
        # 验证图像
        self.assertIsInstance(image, Image.Image, "返回的图像应该是PIL.Image类型")
        self.assertEqual(len(image.size), 2, "图像应该是2D的")
        self.assertEqual(image.mode, "RGB", "图像应该是RGB模式")
        
        # 验证描述
        self.assertIsInstance(description, str, "返回的描述应该是字符串类型")
        self.assertGreater(len(description), 0, "描述不应该为空")
        
    def test_dataloader(self):
        """测试数据加载器"""
        dataset = DTDDataset(self.root_dir, split="train")
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn
        )
        
        # 获取一个批次
        batch = next(iter(dataloader))
        
        # 验证批次格式
        self.assertIn("images", batch, "批次应该包含images键")
        self.assertIn("descriptions", batch, "批次应该包含descriptions键")
        self.assertEqual(len(batch["images"]), self.batch_size, "图像数量应该等于batch_size")
        self.assertEqual(len(batch["descriptions"]), self.batch_size, "描述数量应该等于batch_size")
        
    def test_visualization(self):
        """测试数据可视化"""
        dataset = DTDDataset(self.root_dir, split="train")
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn
        )
        
        # 获取一个批次并可视化
        batch = next(iter(dataloader))
        
        plt.figure(figsize=(15, 5))
        for i in range(self.batch_size):
            plt.subplot(1, self.batch_size, i + 1)
            plt.imshow(batch["images"][i])
            plt.title(batch["descriptions"][i], fontsize=8)
            plt.axis('off')
        
        # 保存可视化结果
        os.makedirs("tests/outputs", exist_ok=True)
        plt.savefig("tests/outputs/dtd_samples.png")
        plt.close()
        
        self.assertTrue(os.path.exists("tests/outputs/dtd_samples.png"), 
                       "可视化结果应该被保存")

if __name__ == "__main__":
    unittest.main() 