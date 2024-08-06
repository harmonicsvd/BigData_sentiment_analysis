### Base & Package Versions
FROM ubuntu:24.04
USER root
ENV PYTHON_VERSION=3.9.6
ENV KAFKA_VERSION=3.1.0
ENV SCALA_VERSION=2.12
ENV KAFKA_PYTHON=2.0.2
ENV CONFLUENT_KAFKA_VERSION=2.3.0
ENV PYFLINK_VERSION=1.0
ENV FLINK_VERSION=1.17.2
ENV PANDAS_VERSION=2.2.2
ENV TORCH_VERSION=2.0.1
ENV NUMPY_VERSION=1.23.5
ENV DASH_VERSION=2.9.3
ENV DASH_BOOTSTRAP_VERSION=1.6.0
ENV PLOTLY_VERSION=5.22.0
ENV FLASK_VERSION=3.0.3
ENV FLOWER_VERSION=1.9
ENV NLTK_VERSION=3.6.5
ENV SCIKIT_VERSION=1.5.0
ENV TRANSFORMERS_VERSION=4.42.2
ENV DOTENV_VERSION=1.0.1





### Required Packages and Miniconda installation
RUN apt update && \
    apt install -y \
    wget \
    curl \
    openjdk-11-jdk-headless \
    neofetch \
    netcat-traditional \
    telnet \
    jq \
    nano \
    vim \
    iputils-ping \
    apt-transport-https \
    ca-certificates \
    gnupg

RUN mkdir -p ~/miniconda3  && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash && \
    ~/miniconda3/bin/conda init zsh
    




### Install Apache Kafka
ENV KAFKA_HOME=/opt/kafka
ENV ZOO_CFG_EXTRA="4lw.commands.whitelist=ruok"
ENV PATH=$PATH:$KAFKA_HOME/bin
ENV KAFKA_LOG_DIRS=/tmp/kafka-logs
RUN curl -q https://archive.apache.org/dist/kafka/$KAFKA_VERSION/kafka_$SCALA_VERSION-$KAFKA_VERSION.tgz -o /tmp/kafka.tgz && \
    tar xvzf /tmp/kafka.tgz -C /opt && \
    mv /opt/kafka_$SCALA_VERSION-$KAFKA_VERSION/ /opt/kafka && \
    rm -rf /tmp/kafka.tgz && \
    chmod -R 777 /opt/kafka && \
    echo "4lw.commands.whitelist=*" >> /opt/kafka/config/zookeeper.properties





### Install Apache Flink, flink-kafka-connector
ENV FLINK_HOME=/opt/flink
ENV PATH=$PATH:$FLINK_HOME/bin
RUN curl -q https://archive.apache.org/dist/flink/flink-$FLINK_VERSION/flink-$FLINK_VERSION-bin-scala_$SCALA_VERSION.tgz -o /tmp/flink.tgz && \
    tar xvzf /tmp/flink.tgz -C /opt && \
    mv /opt/flink-1.17.2/ /opt/flink && \
    rm -rf /tmp/flink.tgz && \
    chmod -R 777 /opt/flink
COPY learning/flink-conf.yaml /opt/flink/conf/
#RUN chmod 777 /opt/flink/conf/flink-conf.yaml    
RUN curl -q https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/$FLINK_VERSION/flink-sql-connector-kafka-$FLINK_VERSION.jar -o /flink-sql-connector-kafka-$FLINK_VERSION.jar





### Port expose
# 9092:kafka
# 2181:zookeeper
# 8081:flink
# 8000:flink dashboard
# 5000:UI
EXPOSE 9092 8081 2181 8000 5000





### Uploading local scripts, modifying to be executable
RUN mkdir learning && mkdir bash && \
    mkdir learning/dash_app && mkdir learning/templates && mkdir learning/trainedmodel

COPY bash/init.sh /bash/init.sh
COPY bash/conda.sh /bash/conda.sh
COPY bash/trigger.sh /bash/trigger.sh
COPY bash/pip_dependencies.sh /bash/pip_dependencies.sh
COPY bash/healthcheck.sh /bash/healthcheck.sh

COPY learning/dash_app/input.py /learning/dash_app/input.py

COPY learning/templates/index.html /learning/templates/index.html

COPY learning/trainedmodel/datasource.json /learning/trainedmodel/datasource.json
COPY learning/trainedmodel/model1.pth /learning/trainedmodel/model1.pth
COPY learning/trainedmodel/sentiment_lstm_model.pth /learning/trainedmodel/sentiment_lstm_model.pth
COPY learning/trainedmodel/sentiment_lstm_model2.pth /learning/trainedmodel/sentiment_lstm_model2.pth
COPY learning/trainedmodel/sentiment_lstm_model3.pth /learning/trainedmodel/sentiment_lstm_model3.pth
COPY learning/trainedmodel/sentiment_lstm_model4.pth /learning/trainedmodel/sentiment_lstm_model4.pth
COPY learning/trainedmodel/smodel.pth /learning/trainedmodel/smodel.pth
COPY learning/trainedmodel/vec.pkl /learning/trainedmodel/vec.pkl


COPY learning/app.py /learning/app.py
COPY learning/client.py /learning/client.py
COPY learning/flinkjob.py /learning/flinkjob.py
COPY learning/gmodel.py /learning/gmodel.py
COPY learning/KProducer.py /learning/KProducer.py
COPY learning/predict.py /learning/predict.py
COPY learning/server.py /learning/server.py
COPY learning/testconsumer.py /learning/testconsumer.py
COPY learning/train.py /learning/train.py

RUN chmod -R 777 bash/ && \
    chmod -R 777 learning/





### Start
CMD ["/bin/sh", "-c", "/bash/init.sh"]
