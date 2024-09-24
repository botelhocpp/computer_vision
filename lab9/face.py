# Detecção de rostos em imagens utilizando três diferentes métodos:
# * Haar Cascade
# * YOLOv8
# * RetinaFace
# Calcula a métrica Intersection over Union (IoU) para comparar as 
# caixas delimitadoras (bounding boxes) detectadas por esses métodos. 

import cv2
import numpy as np
from retinaface import RetinaFace
from ultralytics import YOLO
import os

# Calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate interesection coordinates
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # Calculate interesection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Bounding boxes area
    box1Area = w1 * h1
    box2Area = w2 * h2

    # Calculate IoU
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

output_directory = "output/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load Haar Cascade from the donwloaded file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-treined YOLOv8 for face detection
model_yolo = YOLO("yolov8n.pt")  # YOLOv8 Nano

# Detect faces and draw bounding boxes
def detect_and_draw_faces(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HAAR Cascade
    faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4)

    # YOLOv8
    results = model_yolo(img)
    faces_yolo = results[0].boxes.xywh.cpu().numpy()
    faces_yolo = [[int(x - w/2), int(y - h/2), int(w), int(h)] for (x, y, w, h) in faces_yolo]

    # RetinaFace
    faces_retina = RetinaFace.detect_faces(img_path)
    faces_retina = [[int(faces_retina[key]['facial_area'][0]), int(faces_retina[key]['facial_area'][1]),
                     int(faces_retina[key]['facial_area'][2] - faces_retina[key]['facial_area'][0]),
                     int(faces_retina[key]['facial_area'][3] - faces_retina[key]['facial_area'][1])]
                    for key in faces_retina]

    # Draw bounding boxes in images
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for Haar Cascade

    for (x, y, w, h) in faces_yolo:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for YOLO

    for (x, y, w, h) in faces_retina:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for RetinaFace

    # Save images w/ bounding boxes
    output_path = os.path.join(output_directory, f"output_{os.path.basename(img_path)}")
    cv2.imwrite(output_path, img)

    # Calcular IoU entre os métodos
    iou_haar_yolo = 0
    iou_haar_retina = 0
    num_comparisons_yolo = 0
    num_comparisons_retina = 0

    # Compare Haar w/ YOLO
    for box1 in faces_haar:
        for box2 in faces_yolo:
            iou_haar_yolo += calculate_iou(box1, box2)
            num_comparisons_yolo += 1

    # Comapre Haar w/ RetinaFace
    for box1 in faces_haar:
        for box2 in faces_retina:
            iou_haar_retina += calculate_iou(box1, box2)
            num_comparisons_retina += 1

    # Calculate IoU average
    avg_iou_haar_yolo = iou_haar_yolo / num_comparisons_yolo if num_comparisons_yolo > 0 else 0
    avg_iou_haar_retina = iou_haar_retina / num_comparisons_retina if num_comparisons_retina > 0 else 0

    print(f"Average of IoU Haar vs YOLO for {img_path}: {avg_iou_haar_yolo}")
    print(f"Average of IoU Haar vs RetinaFace for {img_path}: {avg_iou_haar_retina}")

image_paths = [
    '../images/jerry1.jpg',
    '../images/jerry2.jpg',
    '../images/people.jpg'
]

# Process each image
for img_path in image_paths:
    print(f"Processing {img_path}...")
    detect_and_draw_faces(img_path)
