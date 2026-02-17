# FedPG-BR Docker Image
# Multi-purpose image for server, clients, and dashboard

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dashboard]"

# Expose ports
# 8080 - Flower server
# 5000 - Web dashboard
EXPOSE 8080 5000

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "fedpg_br.deploy_server", "--host", "0.0.0.0"]
