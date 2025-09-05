# file: bogie_counter_leading_edge.py
import cv2
from ultralytics import YOLO

# --- YOUR PATHS ---
VIDEO_PATH = "videos/train_tracks.mp4"
MODEL_PATH = "model_bogie_detect.pt"
BOGIE_CLASS_ID = 0  # update if your dataset uses a different index

# --- COUNTER SETTINGS ---
LINE_X = 900             # vertical counting line in pixels
BAND = 12                # hysteresis band around the line (pixels)
PERSIST_FRAMES = 2       # frames the leading edge must remain beyond the line to confirm
IMGSZ = 640
CONF = 0.35
COUNT_DIRECTION = "either"   # "lr" (left->right), "rl" (right->left), or "either"

# If a box is huge and the line sits inside it for many frames, you can instead count
# on the first frame where the line becomes inside the box.
USE_INTERSECT_FALLBACK = False  # set True if you prefer "intersect once" logic

# ------------------------------------------
def beyond(a, thresh, dirn):
    """Is value a beyond threshold in the given direction with band hysteresis?"""
    if dirn == "lr":   # moving right, check right-side beyond line + band
        return a >= thresh + BAND
    else:              # "rl"
        return a <= thresh - BAND

def dir_ok(dirn):
    if COUNT_DIRECTION == "either":
        return True
    return dirn == COUNT_DIRECTION

# ------------------------------------------
model = YOLO(MODEL_PATH)

# Per-track state
prev_cx = {}               # previous center x (to estimate motion direction)
prev_x1, prev_x2 = {}, {}  # previous bbox edges
cross_progress = {}        # tid -> {"dir": "lr"/"rl", "frames": int}
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
    frame = r.orig_img
    annotated = r.plot()

    # Draw line & band
    h, w = frame.shape[:2]
    LINE_X_CLAMP = max(0, min(LINE_X, w - 1))
    cv2.line(annotated, (LINE_X_CLAMP, 0), (LINE_X_CLAMP, h), (0, 0, 255), 2)
    cv2.rectangle(annotated, (LINE_X_CLAMP - BAND, 0), (LINE_X_CLAMP + BAND, h), (0, 0, 255), 1)

    if r.boxes is not None and r.boxes.id is not None:
        ids = r.boxes.id.int().tolist()
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.cpu().numpy(), ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx = (x1 + x2) // 2

            # Visual aids
            cv2.circle(annotated, (cx, (y1 + y2)//2), 4, (0, 255, 255), -1)
            cv2.putText(annotated, f"id {tid}", (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if tid in counted_ids:
                continue

            # Estimate motion direction from center x delta
            prevcx = prev_cx.get(tid, cx)
            dx = cx - prevcx
            if abs(dx) < 1:
                # no strong evidence this frame; reuse previous direction if any
                dirn = cross_progress.get(tid, {}).get("dir")
            else:
                dirn = "lr" if dx > 0 else "rl"

            # Leading edge logic:
            # - If moving right: watch RIGHT edge (x2) cross from left side to beyond the line.
            # - If moving left:  watch LEFT edge  (x1) cross from right side to beyond the line.
            pvx1 = prev_x1.get(tid, x1)
            pvx2 = prev_x2.get(tid, x2)

            # 1) INTERSECT-ONCE fallback (optional)
            if USE_INTERSECT_FALLBACK:
                prev_inside = (pvx1 <= LINE_X_CLAMP <= pvx2)
                curr_inside = (x1   <= LINE_X_CLAMP <= x2)
                # Count on first entry into intersection zone in the allowed direction
                if not prev_inside and curr_inside and dirn and dir_ok(dirn):
                    # start/confirm persistence
                    prog = cross_progress.get(tid, {"dir": dirn, "frames": 0})
                    if prog["dir"] != dirn:
                        prog = {"dir": dirn, "frames": 0}
                    prog["frames"] += 1
                    cross_progress[tid] = prog
                    if prog["frames"] >= PERSIST_FRAMES:
                        bogie_count += 1
                        counted_ids.add(tid)
                        cross_progress.pop(tid, None)

            # 2) LEADING-EDGE crossing (recommended)
            if dirn and dir_ok(dirn):
                if dirn == "lr":
                    # previously wholly left of line? (right edge left of line - BAND)
                    prev_left = (pvx2 < LINE_X_CLAMP - BAND)
                    # now clearly beyond the line?
                    now_beyond = (x2 >= LINE_X_CLAMP + BAND)
                    if prev_left and now_beyond:
                        prog = cross_progress.get(tid, {"dir": "lr", "frames": 0})
                        if prog["dir"] != "lr":
                            prog = {"dir": "lr", "frames": 0}
                        prog["frames"] += 1
                        cross_progress[tid] = prog
                        if prog["frames"] >= PERSIST_FRAMES:
                            bogie_count += 1
                            counted_ids.add(tid)
                            cross_progress.pop(tid, None)
                else:  # "rl"
                    prev_right = (pvx1 > LINE_X_CLAMP + BAND)
                    now_beyond = (x1 <= LINE_X_CLAMP - BAND)
                    if prev_right and now_beyond:
                        prog = cross_progress.get(tid, {"dir": "rl", "frames": 0})
                        if prog["dir"] != "rl":
                            prog = {"dir": "rl", "frames": 0}
                        prog["frames"] += 1
                        cross_progress[tid] = prog
                        if prog["frames"] >= PERSIST_FRAMES:
                            bogie_count += 1
                            counted_ids.add(tid)
                            cross_progress.pop(tid, None)

            # Update history
            prev_cx[tid] = cx
            prev_x1[tid] = x1
            prev_x2[tid] = x2

    # HUD
    cv2.putText(annotated, f"Bogies: {bogie_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    cv2.imshow("Bogie counter (leading edge)", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
