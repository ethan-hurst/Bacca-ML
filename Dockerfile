# First stage: Use TensorFlow GPU image to get pre-installed TensorFlow
FROM tensorflow/tensorflow:latest-gpu AS tensorflow-base

# Second stage: Use your own image to build the environment
FROM python:3.11-slim AS baccarat-env

# Set the working directory
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install required system packages
RUN apt-get update && apt-get install -y python3-pip python3-venv

# Create a virtual environment
RUN python3 -m venv /app/venv

# Copy TensorFlow from the first stage (tensorflow-base)
COPY --from=tensorflow-base /usr/local/lib/python3.11/dist-packages/tensorflow /app/venv/lib/python3.11/site-packages/tensorflow

# Install other Python dependencies in the virtual environment
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r requirements.txt

# Set the virtual environment in the PATH
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose TensorBoard port
EXPOSE 6006

# Ensure correct Python binary is used from the virtual environment
CMD ["/app/venv/bin/python", "main.py"]
