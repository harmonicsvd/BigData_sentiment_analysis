#!/bin/sh
neofetch



# overriding flink service port
#awk '1; END {print "rest.port: 8000"}' /opt/flink/conf/flink-conf.yaml > /opt/flink/conf/temp-flink-conf.yaml
#chmod 777 /opt/flink/conf/temp-flink-conf.yaml
#mv /opt/flink/conf/temp-flink-conf.yaml /opt/flink/conf/flink-conf.yaml
#awk '1; END {print "rest.bind-port: 8000-8090"}' /opt/flink/conf/flink-conf.yaml > /opt/flink/conf/temp-flink-conf.yaml
#mv /opt/flink/conf/temp-flink-conf.yaml /opt/flink/conf/flink-conf.yaml
#awk '1; END {print "rest.bind-address: 0.0.0.0"}' /opt/flink/conf/flink-conf.yaml > /opt/flink/conf/temp-flink-conf.yaml
#mv /opt/flink/conf/temp-flink-conf.yaml /opt/flink/conf/flink-conf.yaml



printf "\nYou may go inside the container now!\n"



# Keep the script running
tail -f /dev/null
