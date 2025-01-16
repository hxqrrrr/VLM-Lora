from setuptools import setup, find_packages

setup(
    name="vlm_lora",
    version="0.1.0",
    description="Vision-Language Model with LoRA fine-tuning support",
    author="hxqrrrr",
    author_email="",
    url="https://github.com/hxqrrrr/VLM-Lora",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.14.0",
        "datasets>=2.0.0",
        "bitsandbytes>=0.41.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 