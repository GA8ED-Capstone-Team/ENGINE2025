FROM python:3.9-slim

# Install system packages
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Set up Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]