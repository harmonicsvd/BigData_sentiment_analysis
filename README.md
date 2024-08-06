# BD24_Project_A3_A

- Our project is about sentiment analysis based on Reddit comments
- What is Sentiment? Method used in NLP to understand opinions, emotions/attitude expressed towards specific keywords.
- In this project we are using Reddit data stream from the “The pushshift Reddit Dataset”, to train our Federated machine learning model to get “Sentiment” for user provided keywords

## 1: Setting up Google Cloud Instance

This projekt requires a VM Server with working Display Manager (GUI), if this is not being run locally.

### 1.1: Create a Google Cloud VM Server

Open Terminal on [Google Cloud](https://console.cloud.google.com/compute/) console and paste below command, with mentioned 3 changes.

1. Change `--project` from your URL
2. Change `--service-account` from: Top Right 3 Dots -> Project Settings -> Service Accounts
3. Change `device-name` key to `"ubuntu-yourname"` in `--ceate-disk` line

```
gcloud compute instances create ubuntu-heavy-devarshi \
    --project=strong-matrix-422110 \
    --service-account=738032921164-compute@developer.gserviceaccount.com \
    --create-disk=auto-delete=yes,boot=yes,device-name=ubuntu-heavy-devarshi,image=projects/ubuntu-os-cloud/global/images/ubuntu-minimal-2404-noble-amd64-v20240625,mode=rw,size=1024,type=projects/strong-matrix-422110/zones/europe-west3-c/diskTypes/pd-ssd \
    --zone=europe-west3-c \
    --machine-type=c2-standard-8 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --instance-termination-action=STOP \
    --provisioning-model=SPOT \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --enable-display-device \
    --tags=http-server,https-server,lb-health-check \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
```

### 1.2: Install `gcloud` CLI on your local terminal

The VM Server can be accessed using `gcloud`, which needs `gcloud` CLI setup on local terminal, which can be done based on your machine type.

- For macOS, please follow this guide: [Install the gcloud CLI on Mac](https://cloud.google.com/sdk/docs/install#mac)
- For Windows, please follow this guide: [Install the gcloud CLI on Windows](https://cloud.google.com/sdk/docs/install#windows)
- For Debian based Linux, follow:

```
# update the system:
sudo apt update -y && sudo apt upgrade -y

# install required dependencies:
sudo apt install apt-transport-https ca-certificates gnupg curl -y

# get the apt keyring:
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# install gcloud:
sudo apt update -y && sudo apt install google-cloud-cli -y

# initialize the gcloud CLI:
gcloud init

# check connectivity with VM server:
gcloud compute ssh --zone "europe-west3-c" "<instance name>" --project "<your projekt name>"
```

### 1.3: Install Display Manager on VM Server

```
# login to the VM Server:
gcloud compute ssh --zone "europe-west3-c" "ubuntu-heavy-devarshi" --project "strong-matrix-422110"

# install chrome-remote-desktop debian package:
wget https://dl.google.com/linux/direct/chrome-remote-desktop_current_amd64.deb
sudo apt install --assume-yes ./chrome-remote-desktop_current_amd64.deb

# update the system and install SLIM (Display Manager) & Ubuntu Desktop:
sudo apt update && sudo apt upgrade -y
sudo apt install slim ubuntu-desktop -y

# restart the VM Server:
sudo reboot
```

From [Google Cloud](https://console.cloud.google.com/compute/), Stop and Start the VM Server. Then,

```
# login to the VM Server again:
gcloud compute ssh --zone "europe-west3-c" "ubuntu-heavy-devarshi" --project "strong-matrix-422110"

# start the SLIM Display Manager:
sudo service slim start
```

On Chromium based Browser, go to [Chrome Remote Desktop](https://remotedesktop.google.com/headless)

1. Click `Begin`, `Next`, `Autorize`.
2. Copy command mentioned under 'Debian Linux' on the VM Server.
3. Enter a 6-digit PIN when prompted. This PIN will be used when you log into the VM instance from your Chrome.
4. On your local computer, go to the [Chrome Remote Desktop](https://remotedesktop.google.com). You will find your Ubuntu Desktop shows up in the portal.
5. Enter the sdame 6-digit PIN you put in step 3.
6. If a pop-up "Authentication Required", cancel it. And now you are in Display Manager of the VM Server!

## 2: Install Docker

```
# login to the VM Server:
gcloud compute ssh --zone "europe-west3-c" "ubuntu-heavy-devarshi" --project "strong-matrix-422110"

# update the system:
sudo apt update -y && sudo apt upgrade -y

# install Docker:
sudo apt install docker.io docker-buildx jq htop iputils-ping -y

# check whether the current loggedin user has permission to run Docker without being root. If not, allow the same:
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

# login to Docker Hub in order to pull the projekt image:
docker login
```

## 3: Run!

Assuming the you have logged into Dockerhub on the terminal on VM. 
```Note: To build the image from source, and see the logging of each services, consider running from chapter 4 instead.```

This is the image on the Dockerhub: [https://hub.docker.com/repository/docker/devarshikshah/bd24_project_a3_a/tags](https://hub.docker.com/repository/docker/devarshikshah/bd24_project_a3_a/tags)

1. Setup Conda environment and trigger the services:

```
# pull the latest image:
docker pull devarshikshah/bd24_project_a3_a:latest

# run the image to create container:
# 8000:8000 for flink job dashboard
# 5000:5000 for UI dashboard
docker run --privileged --name bd24_project_a3_a -p 8000:8000 -p 5000:5000 devarshikshah/bd24_project_a3_a &

# go inside the container:
docker exec -it bd24_project_a3_a /bin/bash

# setup conda environment using conda script
bash bash/conda.sh

# activate the environment:
conda activate a3

# trigger the services- Zookeeper, Kafka and Flink, Kafka producer, Flink Job, Server Model, Client Models, Global Model, and UI: 
bash bash/trigger.sh
```

2. Port forward has been set to bind port 8000 of the Container to the 8000 port of the VM, which allows Flink job to be observed from Browser. Get the IP address of the container from below command, open the Browser and goto ```http://<container ip>:8000```

```
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' bd24_project_a3_a
```

3. View the front webpage: [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard). Beware that TLS has not been set, hence the page is not in `https`, allow your Browser to load non `https` page as well.

4. Once the page is loaded, put a Text in the Textbox, and hit run. It will query to the global model to find the sentiment of the given input.
- Here, the Graph showcases the sentiment of the given input. For which, the given input is tested on Global model every 1,5 seconds, regardless of the number of iterations.

5. To test for different input, hit ```Stop``` button, and place the new input.

## 4. Build From Source (Developer/Demo Mode!)

1. To build the Docker image from source; setup SSH Key, clone the code, and let the build script to perform the build process!

```
# clone the code:
git clone git@collaborating.tuhh.de:e-19/teaching/bd24_project_a3_a.git

# trigger the build script
# NOTE: this script deletes all the local images, to avoid unclean build!
cd bd24_project_a3_a
bash build.sh --no-cache

# run the container from locally built image:
docker run --privileged --name bd24_project_a3_a -p 8000:8000 -p 5000:5000 devarshikshah/bd24_project_a3_a &

# go inside the container:
docker exec -it bd24_project_a3_a /bin/bash

# setup conda environment using conda script
bash bash/conda.sh

# install pip dependencies:
conda activate a3
bash bash/pip_dependencies.sh
```

2. To observe the logging of the various services running, open multiple tabs/windows of terminal, login to the container, and activate the conda environment on **EACH** tabs/windows:

```
# tab 1: Zookeeper Server:
conda activate a3 && $KAFKA_HOME/bin/zookeeper-server-start.sh /opt/kafka/config/zookeeper.properties

# tab 2: Kafka Server:
conda activate a3 && $KAFKA_HOME/bin/kafka-server-start.sh /opt/kafka/config/server.properties

# tab 3: Flink Cluster:
conda activate a3 && $FLINK_HOME/bin/start-cluster.sh

# after about 5 seconds, healthcheck the services:
conda activate a3 && bash bash/healthcheck.sh

# tab 4: Kafka Producer:
conda activate a3 && python3 /learning/KProducer.py

# tab 5: Flink Job:
conda activate a3 && $FLINK_HOME/bin/flink run -py /learning/flinkjob.py --jarfile=flink-sql-connector-kafka-1.17.2.jar

# checkout http://<container ip>:8000 on the Browser whether the Flink job is submitted.
# (optional) to see the data distribution performed by Flink, run below script in a separate tab, with conda environment activated.
# conda activate a3 && python3 /learning/testconsumer.py

# tab 6: Server Model:
conda activate a3 && python3 /learning/server.py

# tab 7: Client Model 1:
conda activate a3 && python3 /learning/client.py --client_id 1

# tab 8: Client Model 2:
conda activate a3 && python3 /learning/client.py --client_id 2

# tab 9: Client Model 3:
conda activate a3 && python3 /learning/client.py --client_id 3

# tab 10: Client Model 4:
conda activate a3 && python3 /learning/client.py --client_id 4

# tab 11: Global Model:
conda activate a3 && python3 /learning/global.py

# tab 12: Initiate UI:
conda activate a3 && python3 /learning/app.py
```

3. Port forward has been set to bind port 8000 of the Container to the 8000 port of the VM, which allows Flink job to be observed from Browser. Get the IP address of the container from below command, open the Browser and goto ```http://<container ip>:8000```

```
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' bd24_project_a3_a
```

4. View the front webpage: [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard). Beware that TLS has not been set, hence the page is not in `https`, allow your Browser to load non `https` page as well.

5. Once the page is loaded, put a Text in the Textbox, and hit run. It will query to the global model to find the sentiment of the given input.
- Here, the Graph showcases the sentiment of the given input. For which, the given input is tested on Global model every 1,5 seconds, regardless of the number of iterations.

6. To test for different input, hit ```Stop``` button, and place the new input.

## 5. Codebase

This is the tree structure of the codebase:

```
$ tree
.
├── Dockerfile                  
├── README.md                   
├── bash
│   ├── conda.sh                
│   ├── healthcheck.sh          
│   ├── init.sh                 
│   ├── pip_dependencies.sh     
│   └── trigger.sh              
├── build.sh                    
├── learning
│   ├── KProducer.py
│   ├── __pycache__
│   │   └── preprocessor.cpython-39.pyc
│   ├── app.py
│   ├── client.py
│   ├── dash_app
│   │   └── input.py
│   ├── dataextraction.py
│   ├── deletetopic.py
│   ├── flink-conf.yaml
│   ├── flinkjob.py
│   ├── global.py
│   ├── modelvocab.py
│   ├── predict.py
│   ├── preprocessor.py
│   ├── server.log
│   ├── server.py
│   ├── templates
│   │   └── index.html
│   ├── testconsumer.py
│   ├── train.py
│   └── trainedmodel
│       ├── datasource.json
│       ├── model1.pth
│       ├── sentiment_lstm_model.pth
│       ├── sentiment_lstm_model2.pth
│       ├── sentiment_lstm_model3.pth
│       ├── sentiment_lstm_model4.pth
│       └── vec.pkl
└── text  # kept commands for internal purpose
    ├── gcc_setup.txt
    ├── git_commands.txt
    └── myCommands.txt
```

## 6. User Stories, Owners

Though the user stories created by us were divided among the team members, we worked together on almost all of the tasks.

0. Google Cloud Setup: Devarshi

- Chapter 1 has been added for user who does not have Google Cloud Instance running.

1. Dockerfile: Devarshi

- Instead of using existing preconfigured images, we focused on building our own custom image, containerizing the whole pipeline in a single container, with the configuration which we desired.
- The image is build on Ubuntu Linux. Earlier were based on Alpine Linux, and also Flink's own docker image; however, not all the packages were available on those bases, including Conda. Ubuntu Linux gave granular control over the installation and configurations.

2. Bash scripts: Devarshi

- We believe in automation - from building from source, to deploying the entire pipeline! Hence created bash files which handles these tasks efficiently, which otherwise takes rather many comamnds.

3. READEME: Devarshi

- Made with unaware users in mind, allowing pasting the commands blindly and still manage to deploy the entire pipeline.

4. Data cleaning and preprocessing: Gaurang

5. User Interface design and testing: Spoorthi

6. Kafka, Flink, Federated Machine Learning Pipeline: Varad
   -setup of kafka, flink
   -setup of Pytorch federated server
   -Building LSTM Model for sentiment analysis
   -Integrations of complete pipeline
   -Integration of frontend and backend pipeline

## Troubleshooting

If the error of ```No broker available``` comes after running, re-run the commands from this onwards:

```
# tab 2: Kafka Server:
conda activate a3 && $KAFKA_HOME/bin/kafka-server-start.sh /opt/kafka/config/server.properties

# tab 3, 4, and so on.
```