#!/bin/bash

# Define variables for the Docker image name and tag
IMAGE_NAME="testam"
IMAGE_TAG="250109"

# Build the Docker image using the Dockerfile in the current directoryecho "Building Docker image..."
#docker buildx build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker build -f docker/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .

DIR="$(pwd)"
DIR="/home/ciel/ahn/project/TESTAM/"

# Run the Docker containerecho "Running Docker container..."
docker run --rm -it --gpus all -v ${DIR}:/TESTAM --network host -v /dev:/dev \
--ipc host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --name testam1 \
--privileged --cap-add SYS_ADMIN --cap-add SYS_NICE \
${IMAGE_NAME}:${IMAGE_TAG} /bin/bash 
#-c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/TESTAM"
