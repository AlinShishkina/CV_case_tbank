import os
import sys
from pathlib import Path

# Добавляем app в путь
sys.path.append(str(Path(__file__).parent.parent))

from app.utils import create_yolo_dataset_structure, create_dataset_yaml
from scripts.auto_annotate import generate_annotations

def main():
    # Пути к данным
    raw_image_dir = "data/raw_images"  # Исходные изображения
    output_dir = "data/dataset"        # Подготовленный датасет
    yaml_path = "data/dataset.yaml"    # Конфиг YAML
    annotation_dir = "data/annotations" # Автоаннотации
    
    print("Подготовка датасета для YOLOv8...")
    
    # Создание структуры датасета
    train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir = create_yolo_dataset_structure(
        raw_image_dir, output_dir
    )
    
    # Генерация аннотаций
    print("Генерация автоматических аннотаций...")
    generate_annotations("yolov8n.pt", raw_image_dir, annotation_dir)
    
    # Копирование аннотаций в соответствующие директории
    from distutils.dir_util import copy_tree
    copy_tree(annotation_dir, train_lbl_dir)
    copy_tree(annotation_dir, val_lbl_dir)
    
    # Создание YAML конфига
    create_dataset_yaml(yaml_path, os.path.abspath(output_dir))
    
    print("Датасет подготовлен успешно!")
    print(f"YAML конфиг: {yaml_path}")
    print(f"Train images: {len(list(Path(train_img_dir).glob('*')))}")
    print(f"Train labels: {len(list(Path(train_lbl_dir).glob('*')))}")
    print(f"Val images: {len(list(Path(val_img_dir).glob('*')))}")
    print(f"Val labels: {len(list(Path(val_lbl_dir).glob('*')))}")

if __name__ == "__main__":
    main()