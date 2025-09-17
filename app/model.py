from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import torch

class LogoDetector:
    def __init__(self, model_path: str = "weights/best.pt"):
        """
        Инициализация детектора логотипов YOLOv8
        """
        self.model_path = model_path
        self.model = self.load_model()
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
    
    def load_model(self):
        """Загрузка модели YOLOv8"""
        try:
            # Пытаемся загрузить обученную модель
            model = YOLO(self.model_path)
            print(f"Модель {self.model_path} успешно загружена")
            return model
        except Exception as e:
            print(f"Ошибка загрузки модели {self.model_path}: {str(e)}")
            # Загружаем стандартные веса
            try:
                model = YOLO("yolov8n.pt")
                print("Загружены стандартные веса YOLOv8n")
                return model
            except Exception as e2:
                print(f"Не удалось загрузить модель: {str(e2)}")
                raise Exception("Не удалось загрузить модель YOLOv8")
    
    def detect(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Детекция логотипов на изображении с помощью YOLOv8
        """
        start_time = time.time()
        
        try:
            # Выполнение предсказания
            results = self.model(
                image, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        # Извлечение координат bounding box
                        x_min, y_min, x_max, y_max = box.xyxy[0].astype(int)
                        confidence = box.conf[0]
                        class_id = box.cls[0].astype(int)
                        
                        detection = {
                            "x_min": int(x_min),
                            "y_min": int(y_min),
                            "x_max": int(x_max),
                            "y_max": int(y_max),
                            "confidence": float(confidence),
                            "class_id": int(class_id),
                            "class_name": "tbank_logo"
                        }
                        detections.append(detection)
            
            processing_time = time.time() - start_time
            return detections, processing_time
            
        except Exception as e:
            print(f"Ошибка при детекции: {str(e)}")
            return [], time.time() - start_time
