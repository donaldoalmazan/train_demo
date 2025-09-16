# file: bogie_counter_improved.py
import cv2
from ultralytics import YOLO

# --- Inputs ---
VIDEO_SRC  = 0                          # 0 for webcam, or "videos/train_tracks.mp4"
MODEL_PATH = "model_bogie_yolo8n.pt"    # your trained bogie model
BOGIE_CLASS_ID = 0

# --- Geometry (full-frame coordinates) ---
LINE_X  = 300    # vertical counting line (x)
# Horizontal ROI, portion of frame to run detection on
TRACK_Y = 200    # center y of the horizontal strip, adjust as needed
BAND_H  = 160    # strip height (keep as small as reliable)

# --- Inference --- 
# Image size fed to the Yolo model, compression down will reduce details but increases frame rate processed 
IMGSZ = 416      # default 480; 416/320 for faster, must be multiple of 32
# Confidence threshold for when to generate a bounding box
CONF  = 0.20     # default 0.3 (30%), lower (e.g., 0.2) if fast motion causes misses

# --- Counting state ---
prev_cx = {}     # track_id -> previous center-x (FULL frame)
net_count = 0    # L->R increments, R->L decrements

model = YOLO(MODEL_PATH)

# Webcam latency tweaks (ignored for file input)
if VIDEO_SRC == 0:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.release()

# Open a reader so we can crop a horizontal strip *per frame*
cap = cv2.VideoCapture(VIDEO_SRC)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_SRC}")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    y1 = max(0, TRACK_Y - BAND_H // 2)
    y2 = min(H, TRACK_Y + BAND_H // 2)

    roi = frame[y1:y2, :]  # full width, narrow height

    # Per-frame ROI inference + tracking; persist=True keeps ByteTrack state across calls
    results = model.track(
        source=roi,
        tracker="bytetrack.yaml",
        classes=[BOGIE_CLASS_ID],
        conf=CONF,
        imgsz=IMGSZ,
        stream=False,
        persist=True,     # keep tracker state across frames
        verbose=False
    )
    r = results[0]

    annotated = frame.copy()
    # Guides
    cv2.rectangle(annotated, (0, y1), (W, y2), (128, 128, 128), 1) # horizontal ROI band
    cv2.line(annotated, (LINE_X, 0), (LINE_X, H), (0, 0, 255), 2) # vertical counting line

    if r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy().tolist() if r.boxes.conf is not None else [None] * len(xyxy)
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(xyxy)

        for (bx1, by1, bx2, by2), conf, tid in zip(xyxy, confs, ids):
            # Map ROI box back to full-frame by adding y offset
            x1, y1b, x2, y2b = map(int, (bx1, by1 + y1, bx2, by2 + y1))
            cx, cy = (x1 + x2) // 2, (y1b + y2b) // 2

            # Draw box with ID and conf
            cv2.rectangle(annotated, (x1, y1b), (x2, y2b), (0, 255, 0), 2)
            label = f"id {tid if tid is not None else '?'}"
            if conf is not None:
                label += f" {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(12, y1b - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # Draw center point, for visualization only (drawing not used in counting logic)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

            # Need a valid track id for crossing logic
            if tid is None:
                continue

            # STRICT side change (no band): prev vs current relative to LINE_X
            if tid in prev_cx:
                prev_side = 'L' if prev_cx[tid] < LINE_X else 'R'
                curr_side = 'L' if cx < LINE_X else 'R'
                if prev_side != curr_side:
                    if prev_side == 'L' and curr_side == 'R':
                        net_count += 1
                        print(f"ID {tid} L→R | net={net_count}")
                    elif prev_side == 'R' and curr_side == 'L':
                        net_count -= 1
                        print(f"ID {tid} R→L | net={net_count}")
            prev_cx[tid] = cx

    # Display count
    cv2.putText(annotated, f"Bogie Count: {net_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Horizontal ROI - Net crossings", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


