import logging
from typing import List, Tuple
import flwr as fl
import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

global_model = None

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
        
        hidden = hidden[-1]  # Get the last layer's hidden state
        combined = torch.cat((hidden, tfidf_vector), dim=1)
        output = self.fc(combined)
        return output

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> List[np.ndarray]:
        global global_model
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            logging.info(f"Round {rnd}: Aggregated {len(results)} results successfully.")
            state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights)}
            global_model.load_state_dict(state_dict)
            torch.save(global_model.state_dict(), "/learning/trainedmodel/smodel.pth")
        else:
            logging.warning(f"Round {rnd}: Failed to aggregate results. Retrying...")

        return weights

def main():
    global global_model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_dim = len(tokenizer)
    global_model = SentimentLSTM(input_dim=input_dim, hidden_dim=128, output_dim=3, n_layers=3)

    
    server_config = fl.server.ServerConfig(num_rounds=3)

    logging.info("Starting Flower server...")
    fl.server.start_server(
        server_address="localhost:8082",
        
        config=server_config,
        grpc_max_message_length=1000 * 1024 * 1024,
    )

if __name__ == "__main__":
    main()
