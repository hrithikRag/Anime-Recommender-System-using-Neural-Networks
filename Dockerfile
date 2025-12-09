FROM python:3.8-slim

# Set environment variables to prevent Python from writing .pyc files & Ensure Python output is not buffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt setup.py ./
COPY pipeline ./pipeline
COPY application.py ./

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy the rest of the application code
COPY . .

# Debug: Check what files are present
RUN echo "=== Listing files ===" && \
    ls -la && \
    echo "=== Listing pipeline directory ===" && \
    ls -la pipeline/ || echo "Pipeline directory not found"

# Train the model with error handling
RUN python pipeline/training_pipeline.py && \
    echo "Training completed successfully" || \
    (echo "Training failed - check if data files and configs are present" && exit 1)

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run the app
CMD ["python", "application.py"]