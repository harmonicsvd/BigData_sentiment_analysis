printf "Pip dependencies installation...\n"

printf "\nkafka-python==$KAFKA_PYTHON"
pip3 install kafka-python==$KAFKA_PYTHON >/dev/null 2>&1

printf "\npyflink==$PYFLINK_VERSION"
pip3 install pyflink==$PYFLINK_VERSION >/dev/null 2>&1

printf "\napache-flink==$FLINK_VERSION"
pip3 install apache-flink==$FLINK_VERSION >/dev/null 2>&1

printf "\nconfluent-kafka==$CONFLUENT_KAFKA_VERSION"
pip3 install confluent-kafka==$CONFLUENT_KAFKA_VERSION >/dev/null 2>&1

printf "\npandas==$PANDAS_VERSION"
pip3 install pandas==$PANDAS_VERSION >/dev/null 2>&1

printf "\ntorch==$TORCH_VERSION"
pip3 install torch==$TORCH_VERSION >/dev/null 2>&1

printf "\nnumpy==$NUMPY_VERSION"
pip3 install numpy==$NUMPY_VERSION >/dev/null 2>&1

printf "\ndash==$DASH_VERSION"
pip3 install dash==$DASH_VERSION >/dev/null 2>&1

printf "\ndash_bootstrap_components==$DASH_BOOTSTRAP_VERSION"
pip3 install dash-bootstrap-components==$DASH_BOOTSTRAP_VERSION >/dev/null 2>&1

printf "\nplotly==$PLOTLY_VERSION"
pip3 install plotly==$PLOTLY_VERSION >/dev/null 2>&1

printf "\nflask==$FLASK_VERSION"
pip3 install flask==$FLASK_VERSION >/dev/null 2>&1

printf "\nflwr==$FLOWER_VERSION"
pip3 install flwr==$FLOWER_VERSION >/dev/null 2>&1

printf "\nnltk==$NLTK_VERSION"
pip3 install nltk==$NLTK_VERSION >/dev/null 2>&1

printf "\nscikit-learn==$SCIKIT_VERSION"
pip3 install scikit-learn==$SCIKIT_VERSION >/dev/null 2>&1

printf "\ntransformers==$TRANSFORMERS_VERSION"
pip3 install transformers==$TRANSFORMERS_VERSION >/dev/null 2>&1

printf "\npython-dotenv==$DOTENV_VERSION"
pip3 install python-dotenv==$DOTENV_VERSION >/dev/null 2>&1
