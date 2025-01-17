# VLM-LoRA

基于 LoRA 的视觉语言模型微调框架。

----

- [x] 支持llama模型
- [x] 支持clip模型
- [ ] 数据集
- [ ] lora
- [ ] json config配置
- [ ] 支持llava模型
- [ ] 多任务并行
- [ ] 多种lora变体
- [ ] 注意力变体

## 环境配置

1. 创建虚拟环境：
```bash
conda create -n CLIP-LoRA python=3.8
conda activate CLIP-LoRA
```

2. 安装 PyTorch (CUDA 12.1)：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. 安装依赖：
```bash
 pip install -e .
```

## 测试说明

1. TinyLLaMA 测试：
```python
# 运行 TinyLLaMA 测试脚本
python scripts/test_llama.py

# 测试内容：
# - 模型加载
# - 文本生成
# - 注意力机制
# - RoPE 位置编码
```

2. CLIP 测试：
```python
# 运行 CLIP 测试脚本
python scripts/test_clip.py

# 测试内容：
# - 模型加载
# - 图像特征提取
# - 文本特征提取
# - 图文相似度计算
```

## 注意事项

- 确保模型已下载到本地缓存
- 使用 `torch.float16` 以减少显存占用
- 建议使用 `local_files_only=True` 从本地加载模型 