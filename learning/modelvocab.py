import torch

def get_vocab_size(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    vocab_size = state_dict['model']['decoder.embed_tokens.weight'].size(0)
    return vocab_size

if __name__ == "__main__":
    model_path = "/learning/trainedmodel/model1.pth"
    vocab_size = get_vocab_size(model_path)
    print(f"The vocabulary size from the model's state dictionary is: {vocab_size}")
