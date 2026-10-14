from pathlib import Path


class PersonDetector:
    """YOLOv8n limited to the COCO 'person' class."""

    def __init__(self, weights_path, conf=0.35):
        if not Path(weights_path).exists():
            # Check first: ultralytics would otherwise try to download the file.
            raise FileNotFoundError(f"Detector weights not found: {weights_path}")
        from ultralytics import YOLO  # imported here so Density mode does not pay for it
        self.model = YOLO(str(weights_path))
        self.conf = conf

    def detect(self, frame_bgr):
        result = self.model.predict(frame_bgr, classes=[0], conf=self.conf, verbose=False)[0]
        boxes = []
        for (x1, y1, x2, y2), score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(score)))
        return boxes
