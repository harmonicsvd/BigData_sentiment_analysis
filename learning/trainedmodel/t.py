import torch
import torch.nn as nn
from transformers import BertTokenizer

# Load the tokenizer used during pre-training
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_dim = 30267  # Ensure this matches the pre-trained model's vocab size

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)  # Use the correct input_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 100, output_dim)  # Adjust the TF-IDF feature size as needed

    def forward(self, input_ids, attention_mask, tfidf_vector):
        embedded = self.embedding(input_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, attention_mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = lstm_out[torch.arange(lstm_out.size(0)), attention_mask.sum(dim=1) - 1]  # Get the outputs corresponding to the last valid token
        combined = torch.cat((lstm_out, tfidf_vector), dim=1)
        output = self.fc(combined)
        return output

# Load the pre-trained model
pretrained_model_path = '/Users/varadkulkarni/TUHH/SecondSemester/BigData/FS-Analysis-scripts/trainedmodel/model1.pth'
model = SentimentLSTM(input_dim=input_dim, hidden_dim=128, output_dim=3, n_layers=3)
try:
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Pretrained model loaded successfully")
except Exception as e:
    print(f"Failed to load pretrained model: {e}")


# Define some sample input text
sample_text = ["I love this product!", "This is the worst experience ever."]

# Tokenize the input text
encoded_input = tokenizer(sample_text, padding=True, truncation=True, return_tensors='pt', max_length=128)

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

# Assuming you have TF-IDF vectors for the sample input (here we use dummy data for demonstration)
tfidf_vector = torch.randn(len(sample_text), 100)  # Replace with actual TF-IDF vectors

# Ensure the model is in evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(input_ids, attention_mask, tfidf_vector)
    predictions = torch.argmax(outputs, dim=1)

# Print the predictions
for text, prediction in zip(sample_text, predictions):
    print(f"Text: {text}\nSentiment: {prediction.item()}")
