import cv2
import numpy as np
from typing import List
import requests
import os
from pathlib import Path

def decode_image(image_data: bytes) -> np.ndarray:
    """
    Декодирование изображения из bytes в numpy array
    
    Args:
        image_data: Изображение в формате bytes
        
    Returns:
        Декодированное изображение в формате numpy array
    """
    try:
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise Exception(f"Ошибка декодирования изображения: {str(e)}")

def validate_image_format(filename: str) -> bool:
    """
    Проверка формата изображения
    
    Args:
        filename: Имя файла
        
    Returns:
        True если формат поддерживается, иначе False
    """
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    file_ext = Path(filename).suffix.lower()
    return file_ext in supported_formats

def download_weights(url: str, save_path: str) -> bool:
    """
    Загрузка весов модели
    
    Args:
        url: URL для скачивания
        save_path: Путь для сохранения
        
    Returns:
        True если загрузка успешна, иначе False
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return True
    except Exception as e:
        print(f"Ошибка загрузки весов: {str(e)}")
        return False