# Base image with Python and ML tools
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose MLflow UI port (if used)
EXPOSE 5000

# Default command (can be overridden in docker-compose or CI/CD)
CMD ["python", "src/train.py"]
