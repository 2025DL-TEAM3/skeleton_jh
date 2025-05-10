# Dockerfile
FROM nvcr.io/nvidia/cuda:12.5.0-base-ubuntu22.04

WORKDIR /app

# Install system packages and Python
RUN apt update && apt install -y git python3-pip

# Install Python dependencies (freeze them if possible!)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Just a placeholder. Actual code will be cloned into /repo
CMD ["bash"]
