import cv2
import numpy as np
from pathlib import Path
from app.model import LogoDetector
from app.utils import decode_image
import json
from sklearn.metrics import precision_score, recall_score, f1_score

def load_validation_data(validation_dir: str):
    """
    Загрузка валидационных данных
    
    Args:
        validation_dir: Директория с валидационными данными
        
    Returns:
        Список кортежей (image_path, annotations)
    """
    validation_data = []
    validation_path = Path(validation_dir)
    
    for image_path in validation_path.glob("*.jpg"):
        annotation_path = validation_path / f"{image_path.stem}.json"
        
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            validation_data.append((str(image_path), annotations))
    
    return validation_data

def calculate_iou(boxA, boxB):
    """
    Вычисление Intersection over Union (IoU) для двух bounding box
    
    Args:
        boxA: Первый bounding box [x_min, y_min, x_max, y_max]
        boxB: Второй bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        IoU value
    """
    # Определение координатов пересечения
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Вычисление площади пересечения
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Вычисление площадей каждого bounding box
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Вычисление IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def evaluate_detector(detector, validation_data, iou_threshold=0.5):
    """
    Оценка качества детектора
    
    Args:
        detector: Объект детектора
        validation_data: Валидационные данные
        iou_threshold: Порог IoU для определения true positive
        
    Returns:
        Метрики precision, recall, f1
    """
    all_true_positives = []
    all_false_positives = []
    all_false_negatives = []
    
    for image_path, annotations in validation_data:
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        # Детекция
        detections = detector.detect(image)
        
        # Преобразование предсказаний в формат [x_min, y_min, x_max, y_max]
        pred_boxes = []
        for detection in detections:
            bbox = detection.bbox
            pred_boxes.append([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
        
        # Получение ground truth bounding boxes
        gt_boxes = []
        for ann in annotations.get('bboxes', []):
            gt_boxes.append([ann['x_min'], ann['y_min'], ann['x_max'], ann['y_max']])
        
        # Сопоставление предсказаний с ground truth
        matched_gt = set()
        matched_pred = set()
        
        # Поиск true positives
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break
        
        # Подсчет метрик для текущего изображения
        tp = len(matched_pred)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)
        
        all_true_positives.append(tp)
        all_false_positives.append(fp)
        all_false_negatives.append(fn)
    
    # Вычисление итоговых метрик
    total_tp = sum(all_true_positives)
    total_fp = sum(all_false_positives)
    total_fn = sum(all_false_negatives)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def main():
    # Загрузка детектора
    detector = LogoDetector("weights/best.pt")
    
    # Загрузка валидационных данных
    validation_data = load_validation_data("data/validation")
    
    # Оценка качества
    precision, recall, f1 = evaluate_detector(detector, validation_data)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Сохранение результатов
    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()