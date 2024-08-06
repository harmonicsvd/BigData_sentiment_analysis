import json
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
import pickle
import os

# Initialize the NLTK Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Set of stop words
stop_words = set(stopwords.words('english'))

# Initialize the BERT Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize the TF-IDF Vectorizer with a fixed maximum number of features
max_tfidf_features = 100
tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features)

# Define a function to check if the comment body is empty or contains only stop words
def is_valid_comment(comment):
    body = comment['body'].strip()
    if not body:
        return False
    words = body.split()
    return any(word.lower() not in stop_words for word in words)

# Define a processing function
def process_message(comment):
    if not is_valid_comment(comment):
        return None
    
    body = comment['body']

    # 1. Sentiment Analysis using NLTK
    sentiment_scores = sid.polarity_scores(body)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment_label = 2  # positive
    elif compound_score <= -0.05:
        sentiment_label = 0  # negative
    else:
        sentiment_label = 1  # neutral

    try:
        # 2. Transform using pre-fitted TF-IDF
        tfidf_vector = tfidf_vectorizer.transform([body]).toarray()[0]
    except ValueError as e:
        if str(e) == "empty vocabulary; perhaps the documents only contain stop words":
            return None
        else:
            raise

    # 3. Tokenization using BERT
    tokens = bert_tokenizer(body, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

    # Prepare the processed comment
    processed_comment = {
        'sentiment_label': sentiment_label,
        'tfidf_vector': tfidf_vector.tolist(),  # Convert to list for JSON serialization
        'input_ids': tokens['input_ids'].tolist()[0],  # Convert to list for JSON serialization
        'attention_mask': tokens['attention_mask'].tolist()[0]  # Convert to list for JSON serialization
    }
    
    return processed_comment

# Function to fit and save the TF-IDF vectorizer
def fit_and_save_vectorizer(data, save_path):
    all_bodies = [comment['body'] for comment in data]
    tfidf_vectorizer.fit(all_bodies)
    with open(save_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"TF-IDF vectorizer saved at {save_path}")

# Function to load the TF-IDF vectorizer
def load_vectorizer(load_path):
    global tfidf_vectorizer
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print(f"TF-IDF vectorizer loaded from {load_path}")
    else:
        raise FileNotFoundError(f"No vectorizer found at {load_path}")

# Main function for initial setup
def main(train_data_path, vectorizer_save_path):
    # Load and preprocess training data
    with open(train_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Fit and save the TF-IDF vectorizer
    fit_and_save_vectorizer(training_data, vectorizer_save_path)

if __name__ == "__main__":
    train_data_path = '/path/to/your/training_data.json'  # Update this path
    vectorizer_save_path = '/path/to/your/vec.pkl'  # Update this path
    main(train_data_path, vectorizer_save_path)
