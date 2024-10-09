# Run the container
docker run --gpus all -v ${PWD}:/app -w /app -p 6006:6006 -it baccarat-ml bash

## build the container
docker build -t baccarat-ml .

## Install Tensorflow image
docker pull tensorflow/tensorflow:latest-gpu