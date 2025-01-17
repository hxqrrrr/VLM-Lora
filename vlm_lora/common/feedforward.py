from typing import Dict

import torch
import torch.nn as nn

from .abstracts import VLMFeedForward


class FeedForward(nn.Module):
    """前馈神经网络包装器。"""

    def __init__(self, feed_forward: VLMFeedForward) -> None:
        super().__init__()
        self.feed_forward_: VLMFeedForward = feed_forward

    def state_dict(self) -> Dict[str, nn.Module]:
        """获取状态字典。"""
        return self.feed_forward_.state_dict()

    def forward(self, hidden_states: torch.Tensor, input_args) -> torch.Tensor:
        """前向传播。"""
        return self.feed_forward_.forward(hidden_states, input_args) 