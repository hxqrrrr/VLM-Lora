from setuptools import setup, find_packages

setup(
    name="vlm-lora",
    version="0.1.0",
    description="Visual Language Model with LoRA fine-tuning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.15.0",
        "huggingface-hub>=0.19.0",
        "protobuf>=4.25.0",
        "requests>=2.31.0",
        "packaging>=23.0",
        "filelock>=3.13.0",
        "pyyaml>=6.0.0",
        "regex>=2023.0.0",
        "typing-extensions>=4.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 