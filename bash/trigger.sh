bash /bash/pip_dependencies.sh



printf "\n\n\nTrigger services..."
printf "\nZookeeper Server..."
$KAFKA_HOME/bin/zookeeper-server-start.sh /opt/kafka/config/zookeeper.properties >/dev/null 2>&1 &
sleep 5

printf "\nKafka Server..."
$KAFKA_HOME/bin/kafka-server-start.sh /opt/kafka/config/server.properties >/dev/null 2>&1 &
sleep 5

printf "\nFlink Cluster..."
$FLINK_HOME/bin/start-cluster.sh >/dev/null 2>&1 &
sleep 10



bash /bash/healthcheck.sh
sleep 10



printf "\n\n\nInitiate Kafka Producer in background..."
python3 /learning/KProducer.py >/dev/null 2>&1 &
sleep 20

printf "\nInitiate Flink..."
$FLINK_HOME/bin/flink run -py /learning/flinkjob.py --jarfile=flink-sql-connector-kafka-1.17.2.jar >/dev/null 2>&1 &
sleep 10

printf "\nInitiate Server model in background..."
python3 /learning/server.py >/dev/null 2>&1 &
sleep 10

printf "\nInitiate Client models in background..."
python3 /learning/client.py --client_id 1 >/dev/null 2>&1 &
python3 /learning/client.py --client_id 2 >/dev/null 2>&1 &
python3 /learning/client.py --client_id 3 >/dev/null 2>&1 &
python3 /learning/client.py --client_id 4 >/dev/null 2>&1 &
sleep 5

printf "\nInitiate Global model in background..."
python3 /learning/global.py >/dev/null 2>&1 &
sleep 10

printf "\nInitiate UI load in background..."
python3 /learning/app.py >/dev/null 2>&1 &



# Keep the script running
#tail -f /dev/null
