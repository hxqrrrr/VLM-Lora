from transformers import AutoProcessor, AutoModelForImageTextToText
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

processor = AutoProcessor.from_pretrained("Intel/llava-gemma-2b")
model = AutoModelForImageTextToText.from_pretrained("Intel/llava-gemma-2b")