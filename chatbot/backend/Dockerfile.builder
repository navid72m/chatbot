# Dockerfile.builder
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Install PyInstaller
RUN pip install --upgrade pip && pip install pyinstaller

# Copy your code
COPY . .

# Install your requirements
RUN pip install -r requirements.txt

# Build the binary
RUN pyinstaller --onefile main.py --distpath /app/dist
