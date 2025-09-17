FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Создание директорий
RUN mkdir -p weights

# Загрузка YOLOv8n весов
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('YOLOv8n веса успешно загружены')"

EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
