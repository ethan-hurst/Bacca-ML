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


## Ensure that tensorflow is accessible in the virtual environment
docker run -it baccarat-ml /bin/bash
source /app/venv/bin/activate
python -c "import tensorflow as tf; print(tf.__version__)"

## Set up NVIDIA Docker Repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && \
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

## install Nvidia toolkit 
sudo apt-get update
sudo apt-get install -y nvidia-docker2


## Restart docker
sudo systemctl restart docker

## Verify NVIDIA driveres and CUDA are installed 
nvidia-smi


## Verify GPU access in container
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

## Verify CUDA libraries are installed 
dpkg -l | grep cuda
dpkg -l | grep cudnn

## Run TensorBoard
tensorboard --logdir logs/fit/ --port 6007
