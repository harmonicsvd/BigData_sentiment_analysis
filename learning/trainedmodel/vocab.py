from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Print the vocab size
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
