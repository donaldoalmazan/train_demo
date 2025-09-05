# file: train_tracker_file.py
import cv2
from ultralytics import YOLO

VIDEO_PATH = "train_demo.mov"   # replace with your .MOV file path
MODEL_PATH = "yolov8n.pt"       # tiny YOLO model (fastest on Pi); pretrained on COCO dataset
TRAIN_CLASS_ID = 6              # COCO "train" class

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

model = YOLO(MODEL_PATH)

while True:
    ok, frame = cap.read()
    if not ok:
        break  # video ended

    results = model.track(
        source=frame,
        stream=True,
        tracker="bytetrack.yaml",
        classes=[TRAIN_CLASS_ID],
        conf=0.35,
        imgsz=640,   # try 480 if you want more speed
        verbose=False
    )

    for r in results:
        annotated = r.plot()
        cv2.imshow("Train detection + tracking", annotated)

        if cv2.waitKey(20) & 0xFF == 27:  # ESC to quit
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

cap.release()
cv2.destroyAllWindows()
