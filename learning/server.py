import sys
import os
from typing import List
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import pickle
from transformers import BertTokenizer
import numpy as np
from threading import Thread
import time
import logging
from gmodel import main
# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Debug prints
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")



logging.basicConfig(level=logging.INFO)

# Model definition (same as in client.py)
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

# Initialize Flask app for prediction endpoint
app = Flask(__name__)

# Load the TF-IDF vectorizer
with open("learning/trainedmodel/vec.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Global model variable
global_model = None
model_iterations = {}
current_iteration = 0

@app.route('/')
def home():
    return "/learning/templates/index.html"

@app.route('/dashboard/')
def dashboard():
    return send_from_directory('templates', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

@app.route('/predict', methods=['POST'])
def predict():
    global current_iteration, model_iterations, global_model
    if global_model is None:
        logging.error("Global model is not initialized.")
        return jsonify({"error": "Global model is not initialized."}), 500

    data = request.json
    text = data['text']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    tfidf_vector = torch.tensor(tfidf_vectorizer.transform([text]).toarray()[0], dtype=torch.float32)

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        output = global_model(input_ids, attention_mask, tfidf_vector.unsqueeze(0))
        sentiment_value = torch.argmax(output).item()

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment_category = sentiment_map[sentiment_value]

    current_iteration += 1
    if text not in model_iterations:
        model_iterations[text] = []
    model_iterations[text].append(sentiment_value)

    return jsonify({"sentiment": sentiment_category, "value": sentiment_value})

@app.route('/iterations', methods=['GET'])
def get_iterations():
    return jsonify(model_iterations)

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404
@app.route('/update_model', methods=['POST'])
def update_model():
    global global_model
    try:
        state_dict = request.json
        # Convert lists back to tensors
        state_dict = {k: torch.tensor(v) for k, v in state_dict.items()}
        global_model.load_state_dict(state_dict)
        return jsonify({"status": "Model updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def load_global_model():
    global global_model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_dim = tokenizer.vocab_size
    model = SentimentLSTM(input_dim, 128, 3, 3)  # Adjust parameters as necessary
    try:
        model.load_state_dict(torch.load("/learning/trainedmodel/smodel.pth"))
        model.eval()
        logging.info("Global model loaded successfully.")
        global_model = model
    except Exception as e:
        logging.error(f"Error loading global model: {e}")

if __name__ == "__main__":
    flower_thread = Thread(target=main)
    flower_thread.start()

    # Wait for Flower server to initialize global_model
    time.sleep(5)  # Wait for 5 seconds to ensure Flower server starts
    load_global_model()

    # Start Flask server after global_model is initialized
    app.run(host='0.0.0.0', port=5001)
