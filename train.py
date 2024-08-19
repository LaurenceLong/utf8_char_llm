import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import ModelConfig
from data import TextDataset
from model import UTF8CharLLM
from tokenizer import UTF8Tokenizer


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch % 100 == 0:
            print(f'Batch {batch}, Loss: {loss.item():.4f}')
    return total_loss / len(dataloader)


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置配置
    config = ModelConfig(vocab_size=256, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048)

    # 创建tokenizer
    tokenizer = UTF8Tokenizer(config.vocab_size)

    # 创建模型
    model = UTF8CharLLM(**config.to_dict()).to(device)

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 加载数据
    dataset = TextDataset([r'D:\work\utf8_char_llm\data\arithmetic_training_data.jsonl'], tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练循环
    num_epochs = 2
    for epoch in range(num_epochs):
        avg_loss = train(model, dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

    print("Training completed!")


if __name__ == "__main__":
    main()
