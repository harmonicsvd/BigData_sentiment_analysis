from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
import json
import os
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
import pickle
from dotenv import load_dotenv
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Initialize the NLTK Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Initialize the TF-IDF Vectorizer (without fitting it)
tfidf_vectorizer = TfidfVectorizer(max_features=100)

# Set of stop words
stop_words = set(stopwords.words('english'))

# Initialize the BERT Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to check if the comment body is empty or contains only stop words
def is_valid_comment(comment):
    body = comment['body'].strip()
    if not body:
        return False
    words = body.split()
    return any(word.lower() not in stop_words for word in words)

# Define a processing function
def process_message(value):
    comment = json.loads(value)
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
        # 2. Fit and transform using TF-IDF
        with open(os.getenv('TFIDF_VECTORIZER_PATH', '/learning/trainedmodel/vec.pkl'), "rb") as f:
            tfidf_vectorizer = pickle.load(f)
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
    
    return json.dumps(processed_comment)

def main():
    # Set up the execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)  # Set parallelism to 4 to match the number of clients
    env.set_stream_time_characteristic(TimeCharacteristic.ProcessingTime)

    # Kafka configuration
    kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

    # Define Kafka source
    kafka_source = FlinkKafkaConsumer(
        topics=os.getenv('KAFKA_SOURCE_TOPIC', 'raw-reddit-comments'),
        deserialization_schema=SimpleStringSchema(),
        properties={
            'bootstrap.servers': kafka_bootstrap_servers,
            'group.id': os.getenv('FLINK_GROUP_ID', 'flink-consumer-group')
        }
    )

    # Define Kafka sinks for each client
    kafka_sinks = [
        FlinkKafkaProducer(
            topic=f'client{i+1}-topic-data',
            serialization_schema=SimpleStringSchema(),
            producer_config={'bootstrap.servers': kafka_bootstrap_servers}
        ) for i in range(4)
    ]

    # Read from Kafka
    comments_stream = env.add_source(kafka_source)

    # Process stream
    processed_stream = comments_stream \
        .map(lambda x: process_message(x), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .rebalance()  # Rebalance the stream to evenly distribute among partitions

    # Write to Kafka topics
    for i, sink in enumerate(kafka_sinks):
        processed_stream.filter(lambda x, index=i: hash(x) % 4 == index).add_sink(sink)

    # Execute the Flink job
    env.execute("Flink Kafka Processing")

if __name__ == "__main__":
    main()
