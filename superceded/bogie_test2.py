# file: bogie_counter_yolo.py
import cv2
from ultralytics import YOLO

VIDEO_PATH = 0
MODEL_PATH = "model_bogie_yolo8n.pt"
BOGIE_CLASS_ID = 0

LINE_X = 300

IMGSZ = 320
CONF = 0.10

model = YOLO(MODEL_PATH)

prev_x = {}         # track_id -> previous center x
counted_ids = set()
bogie_count = 0

for r in model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    classes=[BOGIE_CLASS_ID],
    stream=True,
    imgsz=IMGSZ,
    conf=CONF,
    verbose=False
):
    frame = r.orig_img.copy()
    annotated = r.plot()

    h, w = frame.shape[:2]
    cv2.line(annotated, (LINE_X, 0), (LINE_X, h), (0, 0, 255), 2)

    # --- NEW: draw centers for ALL detections, even if no ID yet ---
    if r.boxes is not None:
        # xyxy for all boxes
        xyxy = r.boxes.xyxy.cpu().numpy()
        # ids may be None on early frames — handle gracefully
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(xyxy)

        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # draw the center regardless of having an ID yet
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
            label = f"id {tid}" if tid is not None else "id ?"
            cv2.putText(annotated, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Only count once a stable ID exists
            if tid is not None:
                if tid in prev_x:
                    prev_side = "L" if prev_x[tid] < LINE_X else "R"
                    curr_side = "L" if cx < LINE_X else "R"

                    if prev_side != curr_side and tid not in counted_ids:
                        if prev_side == "L" and curr_side == "R":
                            bogie_count += 1
                            print(f"Bogie {tid} L→R, total={bogie_count}")
                        elif prev_side == "R" and curr_side == "L":
                            bogie_count -= 1
                            print(f"Bogie {tid} R→L, total={bogie_count}")

                        counted_ids.add(tid)

                prev_x[tid] = cx

    cv2.putText(annotated, f"Bogies: {bogie_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Bogie counter", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
