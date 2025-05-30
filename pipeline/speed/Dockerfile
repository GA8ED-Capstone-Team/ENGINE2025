FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy camera intrinsics file
COPY camera_intrinsics.npz /app/camera_intrinsics.npz

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "speed_detection.py"] 