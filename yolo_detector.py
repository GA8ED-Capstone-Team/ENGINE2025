
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.classList = ["car", "truck", "bus", "train", "motorcycle"]
        self.confidence = confidence

    def detect(self, image):
        results = self.model.predict(image, conf = self.confidence)
        result = results[0]
        detections = self.make_detections(result)
        return detections
    
    def make_detections(self, result):
        boxes = result.boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = int(x2)-int(x1), int(y2)-int(y1)
            class_number = int(box.cls[0].cpu().numpy())

            if result.names[class_number] not in self.classList:
                continue
            conf = float(box.conf[0].cpu().numpy())
            detections.append((([x1,y1,w,h]), class_number, conf))
        return detections