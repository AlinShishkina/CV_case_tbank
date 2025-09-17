from ultralytics import YOLO
import cv2
import numpy as np
from typing import List
from .utils import download_weights

class LogoDetector:
    def __init__(self, model_path: str = "weights/best.pt"):
        """
        Инициализация детектора логотипов
        
        Args:
            model_path: Путь к файлу с весами модели
        """
        self.model = self.load_model(model_path)
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
    
    def load_model(self, model_path: str):
        """Загрузка модели YOLO"""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise Exception(f"Не удалось загрузить модель: {str(e)}")
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Детекция логотипов на изображении
        
        Args:
            image: Входное изображение в формате numpy array
            
        Returns:
            Список обнаруженных логотипов в формате Detection
        """
        # Выполнение предсказания
        results = self.model(
            image, 
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Извлечение координат bounding box
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    
                    # Создание объекта Detection
                    bbox = BoundingBox(
                        x_min=int(x_min),
                        y_min=int(y_min),
                        x_max=int(x_max),
                        y_max=int(y_max)
                    )
                    
                    detection = Detection(bbox=bbox)
                    detections.append(detection)
        
        return detections