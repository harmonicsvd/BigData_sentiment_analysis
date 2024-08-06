from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
import pandas as pd
import os
import logging
import json
from time import sleep
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kafka Configuration
kafka_config = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    'retries': int(os.getenv('KAFKA_RETRIES', 5)),
    'batch.size': int(os.getenv('KAFKA_BATCH_SIZE', 16384)),
    'linger.ms': int(os.getenv('KAFKA_LINGER_MS', 5)),
    'acks': os.getenv('KAFKA_ACKS', 'all'),
    'compression.type': os.getenv('KAFKA_COMPRESSION_TYPE', 'lz4'),
    'client.id': 'my-producer'
}

# Initialize Kafka Producer
producer = Producer(kafka_config)

# Define topics
topics = [
    {'name': 'raw-reddit-comments', 'num_partitions': 4, 'replication_factor': 1},
]

def create_topics(topics):
    """Create Kafka topics if they don't exist."""
    admin_client = AdminClient(kafka_config)
    try:
        existing_topics = admin_client.list_topics(timeout=10).topics
    except KafkaException as e:
        logger.error(f"Failed to list topics: {e}")
        return

    new_topics = [
        NewTopic(topic['name'], num_partitions=topic['num_partitions'], replication_factor=topic['replication_factor'])
        for topic in topics if topic['name'] not in existing_topics
    ]

    if new_topics:
        fs = admin_client.create_topics(new_topics, operation_timeout=30)
        for topic, f in fs.items():
            try:
                f.result()
                logger.info(f"Topic {topic} created")
            except Exception as e:
                logger.error(f"Failed to create topic {topic}: {e}")
    else:
        logger.info("No new topics to create")

def delivery_report(err, msg):
    """Delivery report callback function called on successful or failed message delivery."""
    if err:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def produce_messages(file_path, sleep_time=0.1, batch_size=100):
    """Read JSON data from file and send each raw comment to Kafka topic."""
    try:
        data = pd.read_json(file_path)
        logger.info(f"Loaded data from {file_path}")

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            for _, row in batch.iterrows():
                comment = row.to_dict()
                raw_comment = json.dumps(comment).encode('utf-8')
                try:
                    producer.produce(topic='raw-reddit-comments', value=raw_comment, callback=delivery_report)
                    logger.info(f"Data passed to 'raw-reddit-comments' topic")
                except (KafkaException, TypeError) as e:
                    logger.error(f"Failed to produce message: {e}")
            producer.poll(sleep_time)
            sleep(sleep_time)

    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        logger.error("No data found in the file.")
    except Exception as e:
        logger.error(f"Error in producing messages: {str(e)}")
    finally:
        producer.flush()

if __name__ == "__main__":
    try:
        file_path = os.getenv('DATA_FILE_PATH', '/learning/trainedmodel/datasource.json')
        sleep_time = float(os.getenv('SLEEP_TIME', 0.1))
        batch_size = int(os.getenv('BATCH_SIZE', 100))
        create_topics(topics)
        produce_messages(file_path, sleep_time, batch_size)
    except KeyboardInterrupt:
        logger.info("Process interrupted")
    finally:
        producer.flush()
