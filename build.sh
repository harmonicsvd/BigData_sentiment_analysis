### Docker Image builder from source

printf "Container list:\n"
docker ps -a



printf "\n\n\nCleanup container...\n"
docker container kill bd24_project_a3_a
docker container rm bd24_project_a3_a
docker container kill $(docker ps -a | awk '{print $1}' | tail -n +2 | tr '\n' ' ')
docker container rm $(docker ps -a | awk '{print $1}' | tail -n +2 | tr '\n' ' ')


printf "\n\n\nImages:\n"
docker image ls



printf "\n\n\nCleanup old image, and pull Ubuntu base image...\n"
docker image rm -f $(docker image ls | awk '{print $3}' | tail -n +2 | tr '\n' ' ')
docker image prune -a -f
docker image rm -f devarshikshah/bd24_project_a3_a
docker pull ubuntu:24.04



if [ -z "$1" ]; then
    printf "\n\n\nBuilding with cache...\n"
    docker build --tag 'devarshikshah/bd24_project_a3_a:latest' . 
elif [ $1 = "--no-cache" ]; then
    printf "\n\n\nBuilding without cache...\n"
    docker build --tag 'devarshikshah/bd24_project_a3_a:latest' . $1
else
    printf "\n\n\nIncorrect flag; please use --no-cache flag to build without cache\n"
fi
cd ..



printf "\n\n\nImages:\n"
docker image ls



printf "\n\n\nRun container & trigger services...\n"
docker run --privileged --name bd24_project_a3_a -p 8000:8000 -p 5000:5000 devarshikshah/bd24_project_a3_a:latest &
