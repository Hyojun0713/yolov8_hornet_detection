from ultralytics import YOLO
import config
import numpy as np

class HornetDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            if cls == 1 and conf > config.CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                w, h = x2 - x1, y2 - y1
                x_center, y_center = x1 + w / 2, y1 + h / 2
                detections.append(([x_center, y_center, w, h], conf, cls))
        return detections