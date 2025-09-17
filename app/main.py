from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import cv2
import numpy as np
from app.model import LogoDetector
from app.utils import decode_image, validate_image_format
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="T-Bank Logo Detection API", version="1.0.0")

# Инициализация детектора
detector = LogoDetector()

# Модели ответов
class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
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
        
        # Проверка времени обработки
        if time.time() - start_time > 8:
            raise HTTPException(status_code=500, detail="Превышено время обработки")
        
        # Детекция логотипов
        detections, processing_time = detector.detect(image)
        
        # Преобразование в формат ответа
        response_detections = []
        for detection in detections:
            bbox = BoundingBox(
                x_min=detection["x_min"],
                y_min=detection["y_min"],
                x_max=detection["x_max"],
                y_max=detection["y_max"]
            )
            response_detections.append(Detection(bbox=bbox))
        
        logger.info(f"Обработка заняла {processing_time:.2f} секунд, найдено {len(response_detections)} логотипов")
        
        return DetectionResponse(detections=response_detections)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy", "model_loaded": detector.model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
