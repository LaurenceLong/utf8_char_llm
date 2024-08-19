# main.py 或 train.py

from model import UTF8CharLLM
from config import ModelConfig

# 使用默认配置
config = ModelConfig()

# 或者自定义某些参数
config = ModelConfig(vocab_size=50000, d_model=768, nhead=12)

model = UTF8CharLLM(config)