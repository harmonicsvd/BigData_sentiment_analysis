import os
from confluent_kafka import Consumer, KafkaException
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def create_consumer(group_id):
    return Consumer({
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    })

def consume_from_topic(consumer, topic, num_messages=10):
    consumer.subscribe([topic])
    
    messages = []
    try:
        while len(messages) < num_messages:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())
            messages.append(json.loads(msg.value().decode('utf-8')))
    finally:
        consumer.close()
    
    return messages

def main():
    num_clients = 4  # Assuming there are 4 clients
    num_messages = 10  # Number of messages to consume from each topic
    
    for i in range(num_clients):
        consumer = create_consumer(f'test-consumer-group-{i+1}')
        topic = f'client{i+1}-topic-data'
        print(f'Consuming from topic: {topic}')
        
        messages = consume_from_topic(consumer, topic, num_messages)
        
        print(f'Messages from {topic}:')
        for message in messages:
            print(message)
        print('\n' + '-'*60 + '\n')

if __name__ == "__main__":
    main()
