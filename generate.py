import torch
from model import UTF8CharLLM
from config import ModelConfig
from tokenizer import UTF8Tokenizer


def generate_text(model, tokenizer, prompt, max_seq_len=100, temperature=0.7, top_k=0, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated = []
    with torch.no_grad():
        for _ in range(max_seq_len):
            outputs, _ = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample the next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated)


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置配置
    config = ModelConfig(vocab_size=256, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048)

    # 创建tokenizer
    tokenizer = UTF8Tokenizer(config.vocab_size)

    # 创建模型
    model = UTF8CharLLM(**config.to_dict()).to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load('path_to_your_trained_model.pth'))

    # 生成文本
    prompt = "从前有座山，"
    generated_text = generate_text(model, tokenizer, prompt, max_seq_len=100, temperature=0.7)

    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()
