import os
import sys
import torch
from ultralytics import YOLO

def main():
    data_config = "data/dataset.yaml"
    epochs = 5                 
    imgsz = 416                 
    batch_size = 16             
    output_dir = "weights"

    print("Начало ускоренного обучения модели YOLOv8...")

    os.makedirs(output_dir, exist_ok=True)

    try:
        model = YOLO("yolov8n.pt")

        results = model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project=output_dir,
            name="tbank_logo_detection_fast",
            optimizer='AdamW',
            lr0=0.001,           
            patience=10,
            save=True,
            save_period=5,
            amp=True               
        )

        print("Обучение завершено успешно!")
        print(f"Веса сохранены в: {output_dir}")

        metrics = model.val()
        print(f"Результаты валидации: mAP50-95: {metrics.box.map:.3f}")

    except Exception as e:
        print(f"Ошибка при обучении: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
