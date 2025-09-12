FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libatlas3-base \
    libopenblas-dev \
    gfortran \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]