# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY sentiment_analysis_tool.py .
COPY api.py .
COPY transcript-schema.json .
COPY transcription-example.json .

# Create directory for models (will be mounted as volume)
RUN mkdir -p /app/.models/llm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose API port
EXPOSE 8000

# Set the entrypoint to the API server by default
ENTRYPOINT ["python", "api.py"]

# No default CMD - runs the API server
