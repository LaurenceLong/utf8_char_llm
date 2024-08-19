from prepare_text import TextDataset
from tokenizer import UTF8Tokenizer

if __name__ == "__main__":
    # Example usage
    tokenizer = UTF8Tokenizer()  # You need to provide your tokenizer here
    dataset = TextDataset('arithmetic_training_data.jsonl', tokenizer, seq_len=256)
    print(f"Dataset length: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample input shape: {x.shape}, Sample output shape: {y.shape}")
