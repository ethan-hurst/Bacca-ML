# Use the official TensorFlow GPU image as a base
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y python3-pip python3-venv

# Create a virtual environment and install Python dependencies
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r requirements.txt

# Copy the local code to the container
COPY . .

# Set the virtual environment in the PATH (force Python to be from venv)
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose TensorBoard port
EXPOSE 6006

# Command to run your Python script using the virtual environment
CMD ["python", "main.py"]
