# Disclaimer: This docker file is provided by ChatGPT

# Step 1: Use an official Python 3.11 base image
FROM python:3.11-slim

# Step 2: Set environment variables to prevent Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ="Asia/Singapore"

# Step 3: Set a working directory
WORKDIR /app

# Step 4: Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Copy the requirements.txt file into the container
COPY requirements.txt .

# Step 6: Install Python dependencies using pip
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Step 7: Copy the rest of the project files into the container
COPY . .

# Step 8: Set the default command to run when starting the container
CMD ["/bin/bash"]
