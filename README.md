# VLM-LoRA

Vision-Language Model with LoRA (Low-Rank Adaptation) fine-tuning support.

## Features

- Support for TinyLLaMA and other vision-language models
- LoRA fine-tuning implementation
- Easy-to-use interface for model adaptation
- Flexible configuration system

## Installation

```bash
# Clone the repository
git clone https://github.com/hxqrrrr/VLM-Lora.git
cd VLM-Lora

# Install dependencies
pip install -e .
```

## Usage

```python
from vlm_lora.common import VLMModelConfig, Prompt
from vlm_lora.models.tinyllama import TinyLLaMAForCausalLM

# Create model configuration
model_config = VLMModelConfig(
    name_or_path_="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
    device_="cuda",
    dtype_="float32"
)

# Create model instance
model = TinyLLaMAForCausalLM(model_config)

# Create a prompt
prompt = Prompt(
    instruction="Translate the following sentence to Chinese",
    input="Hello, world!"
)

# Generate response
response = model.generate(prompt)
print(response)
```

## License

MIT License 