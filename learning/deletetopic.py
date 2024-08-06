from confluent_kafka.admin import AdminClient

# Kafka Configuration
kafka_config = {
    'bootstrap.servers': 'localhost:9092'
}

# Initialize Kafka AdminClient
admin_client = AdminClient(kafka_config)

# Define topics to delete
topics = [ 'client1-topic-data','client2-topic-data','client3-topic-data','client4-topic-data','raw-reddit-comments']

def delete_topics(topics):
    """Delete Kafka topics."""
    fs = admin_client.delete_topics(topics, operation_timeout=30)
    for topic, f in fs.items():
        try:
            f.result()
            print(f"Topic {topic} deleted")
        except Exception as e:
            print(f"Failed to delete topic {topic}: {e}")

if __name__ == "__main__":
    delete_topics(topics)
    print(admin_client.list_topics(timeout=10).topics)
