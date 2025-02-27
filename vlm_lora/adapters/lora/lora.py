import torch
from typing import Optional, Union
from vlm_lora.common.config import LoraConfig

class LoraLayer(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Linear, config: LoraConfig, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = torch.nn.Linear(in_features, config.lora_r_, bias=False, device=self.device)
        self.lora_B = torch.nn.Linear(config.lora_r_, out_features, bias=False, device=self.device)
        self.scaling = config.lora_alpha_ / config.lora_r_
        self.dropout = torch.nn.Dropout(p=config.lora_dropout_)
        
        torch.nn.init.normal_(self.lora_A.weight, std=0.02)
        torch.nn.init.normal_(self.lora_B.weight, std=0.02)
        
        self.use_dora = config.use_dora_
        if self.use_dora:
            self.magnitude_vector = torch.nn.Parameter(torch.ones(out_features, device=self.device))
        else:
            self.magnitude_vector = None
            
        self.to(self.device)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        print(f"输入 x requires_grad: {x.requires_grad}")
        
        base_output = self.base_layer(x)
        print(f"base_output requires_grad: {base_output.requires_grad}")
        
        lora_input = self.dropout(x)
        lora_A_out = self.lora_A(lora_input)
        print(f"lora_A_out requires_grad: {lora_A_out.requires_grad}")
        
        lora_output = self.lora_B(lora_A_out) * self.scaling
        print(f"lora_output requires_grad: {lora_output.requires_grad}")
        
        if self.use_dora and self.magnitude_vector is not None:
            lora_output = lora_output * self.magnitude_vector
            print(f"lora_output (DoRA) requires_grad: {lora_output.requires_grad}")
        
        final_output = base_output + lora_output
        final_output.requires_grad = True
        print(f"final_output requires_grad: {final_output.requires_grad}")
        return final_output