## build the container
docker build -t baccarat-ml .

## Run the container
docker run --gpus all -v ${PWD}:/app -w /app -p 6006:6006 -it baccarat-ml bash

## Install Tensorflow image
docker pull tensorflow/tensorflow:latest-gpu

## Create a virtual environment with access to system packages
RUN python3 -m venv /app/venv --system-site-packages && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r requirements.txt

## Activate the virtual environment
source /app/venv/bin/activate
