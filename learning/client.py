import json
import logging
import requests
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from kafka import KafkaConsumer
import flwr as fl

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, data, max_length=128):
        self.data = data
        self.max_length = max_length
        logger.debug(f"Initialized SentimentDataset with {len(data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        tfidf_vector = torch.tensor(item['tfidf_vector'], dtype=torch.float)
        label = torch.tensor(item['sentiment_label'], dtype=torch.long)
        return input_ids, attention_mask, tfidf_vector, label

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 100, output_dim)  # Adjust the TF-IDF feature size as needed

    def forward(self, input_ids, attention_mask, tfidf_vector):
        embedded = self.embedding(input_ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, attention_mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Using the last hidden state
        hidden = hidden[-1]  # Get the last layer's hidden state

        combined = torch.cat((hidden, tfidf_vector), dim=1)
        output = self.fc(combined)
        return output

def pad_sequence(sequence, max_length):
    """Pads or truncates a sequence to the max_length."""
    padded_sequence = sequence + [0] * (max_length - len(sequence))
    return padded_sequence[:max_length]

def custom_collate_fn(batch):
    """Custom collate function for DataLoader to handle padding."""
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    tfidf_vectors = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
    # Determine the maximum sequence length in the batch
    max_length = max(len(seq) for seq in input_ids)
    
    # Pad input_ids and attention_masks to the maximum length
    padded_input_ids = [pad_sequence(seq.tolist(), max_length) for seq in input_ids]
    padded_attention_masks = [pad_sequence(mask.tolist(), max_length) for mask in attention_masks]
    
    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_masks_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
    tfidf_vectors_tensor = torch.stack(tfidf_vectors)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return input_ids_tensor, attention_masks_tensor, tfidf_vectors_tensor, labels_tensor

def load_data_from_kafka(topic, batch_size):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id='flink-consumer-group',
        consumer_timeout_ms=1000000,
        session_timeout_ms=300000,  # Adjust this value
        heartbeat_interval_ms=100000
    )

    data = []
    logger.info(f"Starting to load data from Kafka topic: {topic}")
    try:
        for message in consumer:
            data.append(json.loads(message.value.decode('utf-8')))
            if len(data) >= batch_size:
                break
    except Exception as e:
        logger.error(f"Error loading data from Kafka topic: {e}")
    finally:
        consumer.close()

    logger.info(f"Loaded {len(data)} messages from Kafka topic: {topic}")
    return data

def load_pretrained_model(model, path):
    try:
        model.load_state_dict(torch.load(path))
        logger.info("Pretrained model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
    return model

def train_model(model, data, flower_client):
    batch_size = 32
    n_epochs = 2  # Train for 2 epochs before sending the model to server
    learning_rate = 0.01

    train_dataset = SentimentDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Starting training for {n_epochs} epochs")
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
        logger.info(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Send updated model to server every 2 epochs
    if flower_client is not None:
        flower_client.send_model()
        logger.info("Model sent to server")
    logger.info("Training completed")
    return model

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    logger.info("Starting evaluation")
    with torch.no_grad():
        for input_ids, attention_mask, tfidf_vector, labels in data_loader:
            outputs = model(input_ids, attention_mask, tfidf_vector)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Evaluation accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    return accuracy, total

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, server_address):
        super(FlowerClient, self).__init__()
        self.model = model
        self.train_data = train_data
        self.server_address = server_address

    def get_parameters(self):
        return {k: v.cpu().detach().numpy().tolist() for k, v in self.model.state_dict().items()}
    
    def set_parameters(self, parameters):
        params_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(params_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_model(self.model, self.train_data, self)
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        train_loader = DataLoader(SentimentDataset(self.train_data), batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
        accuracy, total = evaluate_model(self.model, train_loader)
        return float(accuracy), len(self.train_data), {"accuracy": float(accuracy)}

    def send_model(self):
        logger.info("Sending model to server...")
        params_to_send = self.get_parameters()
        try:
            endpoint_url = "http://localhost:5001/update_model"
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    response = requests.post(endpoint_url, json=params_to_send)
                    if response.status_code == 200:
                        logger.info("Model sent to server successfully")
                        return True
                    else:
                        logger.error(f"Failed to send model to server. Status code: {response.status_code}, response: {response.json()}")
                except requests.RequestException as e:
                    logger.error(f"RequestException while sending model to server: {e}")

                delay = 5 * (2 ** attempt)
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} in {delay} seconds...")
                time.sleep(delay)

            logger.error("Failed to send model to server after multiple attempts")
        except Exception as e:
            logger.error(f"Error sending model to server: {e}")
        return False

def main(client_id):
    topic = f'client{client_id}-topic-data'
    batch_size = 100  # Adjust this based on the desired batch size for Kafka data consumption
    logger.info(f"Client {client_id} starting data load from topic: {topic}")
    
    # Set the input_dim to the vocab size of the pretrained model
    input_dim = 30522  # Ensure this matches the pretrained model's vocab size
    hidden_dim = 128
    output_dim = 3  # Positive, Negative, Neutral
    n_layers = 3

    model = SentimentLSTM(input_dim, hidden_dim, output_dim, n_layers)
    
    # Path to the pre-trained model
    pretrained_model_path = '/learning/trainedmodel/smodel.pth'  # Update with your path
    model = load_pretrained_model(model, pretrained_model_path)
    
    if not model:
        logger.error(f"Model could not be loaded for client {client_id}")
        return
    
    # Flower client setup
    server_address = 'localhost:8082'  # Replace with your server address
    flower_client = FlowerClient(model, None, server_address)  # Initialize Flower client with server address

    while True:
        data = load_data_from_kafka(topic, batch_size)
        if not data:
            logger.info(f"No more data available for client {client_id} from topic: {topic}")
            break

        # Update the Flower client with the new data
        flower_client.train_data = data

        model = train_model(model, data, flower_client)

        train_dataset = SentimentDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
        accuracy, total = evaluate_model(model, train_loader)

        logger.info(f"Client {client_id}: Batch evaluation accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Client Script for Federated Learning with Kafka')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID for the federated learning client')
    args = parser.parse_args()

    main(args.client_id)
