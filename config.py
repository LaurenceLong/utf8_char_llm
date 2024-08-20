from dataclasses import dataclass, asdict
from typing import Dict, Any

from tokenizer.utf8_tokenizer import DEFAULT_VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        self.vocab_size = max(DEFAULT_VOCAB_SIZE, self.vocab_size)
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 创建一个默认配置实例
default_config = ModelConfig()
