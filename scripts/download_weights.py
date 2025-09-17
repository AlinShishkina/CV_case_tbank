import os
from pathlib import Path
from app.utils import download_weights

def main():
    # URL для скачивания весов (замените на реальный URL)
    weights_url = "https://example.com/tbank_yolov8n_weights.pt"
    weights_path = "weights/best.pt"
    
    # Создание директории если не существует
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    # Загрузка весов
    print("Загрузка весов модели...")
    success = download_weights(weights_url, weights_path)
    
    if success:
        print(f"Веса успешно загружены в {weights_path}")
    else:
        print("Не удалось загрузить веса модели")

if __name__ == "__main__":
    main()