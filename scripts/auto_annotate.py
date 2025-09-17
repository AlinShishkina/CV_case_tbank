import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml

def generate_annotations(model_path: str, image_dir: str, output_dir: str, conf_threshold: float = 0.3):
    """
    Автоматическая генерация аннотаций в формате YOLO
    """
    # Загрузка модели
    model = YOLO(model_path)
    
    # Создание директории для аннотаций
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Поиск всех изображений
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    
    for ext in image_extensions:
        images.extend(Path(image_dir).glob(f"*{ext}"))
        images.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    print(f"Найдено {len(images)} изображений для аннотации")
    
    for img_path in images:
        # Загрузка изображения
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Детекция
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Создание файла аннотации
        annotation_path = Path(output_dir) / f"{img_path.stem}.txt"
        
        with open(annotation_path, 'w') as f:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Конвертация в YOLO формат
                        x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2 / image.shape[1]
                        y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2 / image.shape[0]
                        width = (box.xyxy[0][2] - box.xyxy[0][0]) / image.shape[1]
                        height = (box.xyxy[0][3] - box.xyxy[0][1]) / image.shape[0]
                        
                        # Запись в файл: class_id x_center y_center width height
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Аннотация создана: {annotation_path}")

def main():
    # Конфигурация
    model_path = "yolov8n.pt"  # Предобученная модель для начальной аннотации
    image_dir = "data/raw_images"
    output_dir = "data/annotations"
    
    print("Начало автоматической генерации аннотаций...")
    generate_annotations(model_path, image_dir, output_dir)
    print("Генерация аннотаций завершена!")

if __name__ == "__main__":
    main()