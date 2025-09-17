import cv2
import numpy as np
from typing import List
import os
from pathlib import Path
import yaml
import shutil

def decode_image(image_data: bytes) -> np.ndarray:
    """Декодирование изображения из bytes в numpy array"""
    try:
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Ошибка декодирования изображения: {str(e)}")
        return None

def validate_image_format(filename: str) -> bool:
    """Проверка формата изображения"""
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    file_ext = Path(filename).suffix.lower()
    return file_ext in supported_formats

def create_yolo_dataset_structure(base_dir: str, output_dir: str):
    """
    Создание структуры датасета для YOLO
    """
    # Создание директорий
    train_image_dir = Path(output_dir) / "train" / "images"
    train_label_dir = Path(output_dir) / "train" / "labels"
    val_image_dir = Path(output_dir) / "val" / "images"
    val_label_dir = Path(output_dir) / "val" / "labels"
    
    for dir_path in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Поиск всех изображений
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(Path(base_dir).glob(f"**/*{ext}"))
        all_images.extend(Path(base_dir).glob(f"**/*{ext.upper()}"))
    
    # Разделение на train/val (80/20)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Копирование изображений
    for img_path in train_images:
        target = train_image_dir / img_path.name
        if not target.exists():
            shutil.copy2(img_path, target)
    
    for img_path in val_images:
        target = val_image_dir / img_path.name
        if not target.exists():
            shutil.copy2(img_path, target)
    
    print(f"Создано {len(train_images)} train и {len(val_images)} val изображений")
    return str(train_image_dir), str(train_label_dir), str(val_image_dir), str(val_label_dir)

def create_dataset_yaml(output_path: str, data_dir: str):
    """
    Создание YAML файла конфигурации датасета
    """
    config = {
        'path': data_dir,
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'tbank_logo'},
        'nc': 1
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YAML конфиг создан: {output_path}")
