# file: bogie_counter_yolo.py
import cv2
from ultralytics import YOLO

VIDEO_PATH = 0      # "videos/train_tracks.mp4" or 0 for webcam
MODEL_PATH = "model_bogie_detect.pt" # Custom trained model based on YOLOv8n
BOGIE_CLASS_ID = 0            # Model only has one class, 'bogie' with ID 0

LINE_X = 300                  # Vertical counting line x-position, 300 for webcam, ~900 for video
COUNT_DIRECTION = "either"    # "lr" (left->right), "rl" (right->left), or "either"
LINE_BAND = 8                 # Width of the line band to count within, in pixels
PERSIST_FRAMES = 3            # frames a track must remain on the new side to confirm a crossing

IMGSZ =  288                  # Inference size (pixels), 640 is default, try 480 or 320 for faster; must be multiple of 32
CONF = 0.35                   # Confidence threshold, creates a bounding box when above this value

model = YOLO(MODEL_PATH)

# State for counting
prev_x = {}            # track_id -> previous center x
counted_ids = set()    # track_ids that have been counted
bogie_count = 0

# NOTE: calling model.track() ONCE keeps the tracker state persistent across frames.
for r in model.track(
    source=VIDEO_PATH,           # video path or 0 for webcam
    tracker="bytetrack.yaml",
    classes=[BOGIE_CLASS_ID],
    stream=True,
    imgsz=IMGSZ,
    conf=CONF,
    verbose=False
):
    frame = r.orig_img.copy()
    annotated = r.plot()

    # Draw the vertical counting line
    h, w = frame.shape[:2]
    cv2.line(annotated, (LINE_X, 0), (LINE_X, h), (0, 0, 255), 2)

    # Process detections and tracks for each frame
    if r.boxes is not None and r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.cpu().numpy(), ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # draw the center for visualization
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(annotated, f"id {tid}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # if we already counted this track, skip
            if tid in counted_ids:
                continue

            # check for crossing: previous side vs current side relative to LINE_X
            if tid in prev_x:
                prev_side = "L" if prev_x[tid] < LINE_X else "R"
                curr_side = "L" if cx < LINE_X else "R"

                crossed = (prev_side != curr_side)

                # optional direction filter
                if crossed:
                    if COUNT_DIRECTION == "lr" and not (prev_x[tid] < LINE_X and cx >= LINE_X):
                        crossed = False
                    elif COUNT_DIRECTION == "rl" and not (prev_x[tid] > LINE_X and cx <= LINE_X):
                        crossed = False

                if crossed:
                    bogie_count += 1
                    counted_ids.add(tid)

            # update history
            prev_x[tid] = cx

    # show counter
    cv2.putText(annotated, f"Bogies: {bogie_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Bogie counter", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to quit
        break

cv2.destroyAllWindows()
