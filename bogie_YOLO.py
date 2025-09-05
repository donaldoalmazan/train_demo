# file: bogie_counter_yolo.py
import cv2
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = 0   # "train_tracks.mp4" or 0 for webcam
MODEL_PATH = "model_bogie_tracking.pt"         # custom bogie-trained model based on YOLOv8n (nano)
BOGIE_CLASS_ID = 0                # There's only one class in this model, "bogie" class (id=1)

CONF_THRESHOLD = 0.35
IMGSZ = 640     # 640 or try 480 if you want more speed

# Counting line position (vertical line at x=…)
LINE_X = 300 # 900 for demo video
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 2

# ----------------------------
# STATE
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

model = YOLO(MODEL_PATH)

counted_ids = set()
bogie_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.track(
        source=frame,
        stream=True,
        tracker="bytetrack.yaml",  # persistent IDs
        classes=[BOGIE_CLASS_ID],  # detects only bogies
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        verbose=False
    )

    for r in results:
        annotated = r.plot()

        if r.boxes is not None:
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
            for box, track_id in zip(r.boxes.xyxy.cpu().numpy(), ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center point

                # Draw center
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

                # Check if this bogie crosses the vertical line
                if abs(cx - LINE_X) < 5:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        bogie_count += 1
                        print(f"Bogie count = {bogie_count}")

        # Draw counting line + count text
        h, w = frame.shape[:2]
        cv2.line(annotated, (LINE_X, 0), (LINE_X, h), LINE_COLOR, LINE_THICKNESS)
        cv2.putText(annotated, f"Bogies counted: {bogie_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Bogie counter", annotated)

        if cv2.waitKey(20) & 0xFF == 27:  # ESC quits
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

cap.release()
cv2.destroyAllWindows()