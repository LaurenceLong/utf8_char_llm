import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import DecoderLayer
from .embedding import EmbeddingLayer
from .output_layer import OutputLayer
from tokenizer import UTF8Tokenizer


class UTF8CharLLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len=256, dropout=0.1):
        super(UTF8CharLLM, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = UTF8Tokenizer(vocab_size)
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.output_layer = OutputLayer(d_model, vocab_size)

    def forward(self, x):
        # x = self.tokenizer.encode(x)  # data processed
        x = self.embedding(x)

        attention_info = []
        for layer in self.layers:
            x, layer_attention_info = layer(x)
            attention_info.append(layer_attention_info)

        logits = self.output_layer(x)
        return logits, attention_info

    def generate(self, prompt, max_seq_len, temperature=1.0, top_k=0, top_p=0.9):
        self.eval()  # 将模型设置为评估模式
        device = next(self.parameters()).device  # 获取模型所在的设备

        # 将提示转换为token
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated = []
        for _ in range(max_seq_len):
            # 准备输入
            if len(input_ids[0]) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            # 前向传播
            with torch.no_grad():
                logits, _ = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

            # 应用top-k采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # 应用top-p (nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 将新token添加到生成的序列中
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 如果生成了结束标记，就停止生成
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # 将生成的token解码为文本
        generated_text = self.tokenizer.decode(generated)
        return generated_text

    def greedy_search(self, prompt, max_seq_len):
        self.eval()
        device = next(self.parameters()).device

        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated = []
        for _ in range(max_seq_len):
            if len(input_ids[0]) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            with torch.no_grad():
                logits, _ = self(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1)

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated)
        return generated_text
