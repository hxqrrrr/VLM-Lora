from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class Prompt:
    instruction: str
    input: Optional[str] = None
    label: Optional[str] = None

    def format_prompt(self) -> str:
        """格式化提示文本"""
        if self.input:
            return f"### Instruction:\n{self.instruction}\n\n### Input:\n{self.input}\n\n### Response:"
        else:
            return f"### Instruction:\n{self.instruction}\n\n### Response:"

@dataclass
class InputData:
    inputs: Any
    labels: Optional[List[Any]] = None 