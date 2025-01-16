from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def identity_checkpoint(module_fn: Callable, *args, **kwargs):
    return module_fn(*args, **kwargs)


CHECKPOINT_CLASSES = {
    "none": identity_checkpoint,
    "torch": torch_checkpoint,
}
