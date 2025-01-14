FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libasound2-dev build-essential python3-opencv libopencv-dev v4l-utils -y && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
