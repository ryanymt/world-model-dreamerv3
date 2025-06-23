# Use an official NVIDIA CUDA runtime as a parent image.
# This tag for CUDA 12.2.2 is verified to be available.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Install Python 3.11, pip, and other system dependencies.
# xvfb is needed for rendering environments like Atari without a physical display.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip xvfb && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# Install the Python dependencies using the specific python interpreter's pip module.
# This is a more robust way to install packages for a specific python version.
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# The entrypoint.sh script is used to run the application.
# We need to make it executable.
RUN chmod +x ./entrypoint.sh

# Set the entrypoint for the container. When the container starts,
# it will execute this script.
ENTRYPOINT ["./entrypoint.sh"]
