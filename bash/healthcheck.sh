printf "\n\n\nHealthcheck..."
printf "\nZookeeper?:\n"
if echo "ruok" | nc localhost 2181; then
    printf "\n>Zookeeper is healthy."
else
    printf "\n>Zookeeper health check failed."
fi

printf "\n\n\nKafka?:\n"
if $KAFKA_HOME/bin/kafka-topics.sh --list --bootstrap-server localhost:9092; then
    printf "\n>Kafka is healthy."
else
    printf "\n>Kafka health check failed."
fi

printf "\n\n\nFlink?:\n"
if curl -s http://localhost:8000/overview | jq; then
    printf "\n>Flink is healthy."
else
    printf "\n>Flink health check failed."
fi
printf "\n"
