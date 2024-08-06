import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging
import os
import pickle
from preprocessor import process_message, tfidf_vectorizer, max_tfidf_features, is_valid_comment
from sklearn.model_selection import train_test_split
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.data = data
        self.max_length = max_length
        logging.info(f"Dataset initialized with {len(self.data)} entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        tfidf_vector = torch.tensor(item['tfidf_vector'], dtype=torch.float)
        label = torch.tensor(item['sentiment_label'], dtype=torch.long)
        return input_ids, attention_mask, tfidf_vector, label

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

def custom_collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    tfidf_vectors = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
    max_length = max(len(seq) for seq in input_ids)
    input_ids = [pad_sequence(seq.tolist(), max_length) for seq in input_ids]
    attention_masks = [pad_sequence(mask.tolist(), max_length) for mask in attention_masks]
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    tfidf_vectors = torch.stack([torch.tensor(vec.tolist()) for vec in tfidf_vectors])
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, tfidf_vectors, labels

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

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logging.info(f"Model loaded from {path}")
    else:
        logging.warning(f"No model found at {path}. Initializing new model.")
    return model

def train_model(data, model_path=None, continue_training=False):
    # Hyperparameters
    input_dim = max(max(d['input_ids']) for d in data) + 1  
    hidden_dim = 128
    output_dim = 3  # Positive, Negative, Neutral
    n_layers = 3
    batch_size = 32
    n_epochs = 10
    learning_rate = 0.01

    logging.info("Training model with the following hyperparameters:")
    logging.info(f"Input dimension: {input_dim}")
    logging.info(f"Hidden dimension: {hidden_dim}")
    logging.info(f"Output dimension: {output_dim}")
    logging.info(f"Number of layers: {n_layers}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Number of epochs: {n_epochs}")
    logging.info(f"Learning rate: {learning_rate}")

    # Dataset and Dataloader
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)  # Use 50% of data for test
    logging.info(f"Data split into {len(train_data)} training samples and {len(test_data)} test samples.")

    train_dataset = SentimentDataset(train_data)
    test_dataset = SentimentDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Model, Loss, Optimizer
    model = SentimentLSTM(input_dim, hidden_dim, output_dim, n_layers)
    if continue_training and model_path:
        model = load_model(model, model_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for i, (input_ids, attention_mask, tfidf_vector, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, tfidf_vector)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Log every 10 batches
                logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        logging.info(f"Epoch {epoch+1}/{n_epochs} completed.")

    # Save the model
    model_save_path = model_path if model_path else '/learning/trainedmodel/model3.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim  # Save the input dimension
    }, model_save_path)
    
    logging.info(f"Model saved as {model_save_path}")

    return model

def test_model(model, test_data):
    test_dataset = SentimentDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, tfidf_vector, labels in test_loader:
            outputs = model(input_ids, attention_mask, tfidf_vector)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.info(f"Test Accuracy: {accuracy:.4f}")

def fit_and_save_vectorizer(data, save_path):
    all_bodies = [comment['body'] for comment in data]
    tfidf_vectorizer.fit(all_bodies)
    with open(save_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    logging.info(f"TF-IDF vectorizer saved at {save_path}")

def load_vectorizer(load_path):
    global tfidf_vectorizer
    with open(load_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    logging.info(f"TF-IDF vectorizer loaded from {load_path}")

def main(train_new_data=True):
    # Define file paths
    train_file = '/Users/varadkulkarni/TUHH/SecondSemester/BigData/finalscripts/training_data.json/split_13.json'
    test_file = '/Users/varadkulkarni/TUHH/SecondSemester/BigData/finalscripts/training_data.json/split_22.json'
    vectorizer_path = '/Users/varadkulkarni/TUHH/SecondSemester/BigData/FS-Analysis-scripts/trainedmodel/vec.pkl'
    model_path = '/Users/varadkulkarni/TUHH/SecondSemester/BigData/FS-Analysis-scripts/trainedmodel/model1.pth'

    # Load and preprocess training data
    logging.info("Loading training data.")
    with open(train_file, 'r') as f:
        raw_train_data = json.load(f)
    training_data = [comment for comment in raw_train_data if is_valid_comment(comment)]

    # Fit and save the TF-IDF vectorizer if training new data
    if train_new_data:
        fit_and_save_vectorizer(training_data, vectorizer_path)
    
    # Load the fitted tfidf_vectorizer
    load_vectorizer(vectorizer_path)

    # Process training data with the fitted vectorizer
    processed_training_data = []
    for comment in training_data:
        processed_comment = process_message(comment)
        if processed_comment is not None:
            processed_training_data.append(processed_comment)

    logging.info(f"Processed {len(processed_training_data)} training comments.")

    if train_new_data:
        model = train_model(processed_training_data, model_path, continue_training=True)
    else:
        model = SentimentLSTM(len(tfidf_vectorizer.vocabulary_), 128, 3, 3)
        model = load_model(model, model_path)

    # Load and preprocess test data
    logging.info("Loading test data.")
    with open(test_file, 'r') as f:
        raw_test_data = json.load(f)
    test_data = [process_message(comment) for comment in raw_test_data if is_valid_comment(comment)]

    logging.info(f"Processed {len(test_data)} test comments.")
    
    test_model(model, test_data)

if __name__ == "__main__":
    main(train_new_data=True)  # Set to False if you don't want to train on new data
