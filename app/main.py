from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import cv2
import numpy as np
from .model import LogoDetector
from .utils import decode_image, validate_image_format
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="T-Bank Logo Detection API", version="1.0.0")

# Инициализация детектора
detector = LogoDetector()

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении
    """
    start_time = time.time()
    
    # Валидация формата файла
    if not validate_image_format(file.filename):
        raise HTTPException(
            status_code=400, 
            detail="Неподдерживаемый формат изображения. Поддерживаются: JPEG, PNG, BMP, WEBP"
        )
    
    try:
        # Чтение и декодирование изображения
        image_data = await file.read()
        image = decode_image(image_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")
        
        # Детекция логотипов
        detections = detector.detect(image)
        
        # Форматирование результата
        response = DetectionResponse(detections=detections)
        
        logger.info(f"Обработка заняла {time.time() - start_time:.2f} секунд")
        return response
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)