# file: bogie_counter.py - this code doesn't use any AI, simply openCV motion detection. Doesn't work great or scale, this is mostly just a benchmark to tinker with OpenCV image features.
import cv2

VIDEO_PATH = "videos/train_tracks.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

LINE_X = 300   # x-position of vertical counting line
bogie_count = 0
counted_ids = set()

# Background subtractor (for moving wheel/bogie blobs)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = fgbg.apply(gray)
    mask = cv2.medianBlur(mask, 5)

    # Find contours of moving objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # filter noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # Draw box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)

        # Check crossing of vertical line
        if abs(cx - LINE_X) < 5:  # near the line
            bogie_count += 1
            print("Bogie/wheel detected. Count =", bogie_count)

    # Draw line + counter
    cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (255,0,0), 2)
    cv2.putText(frame, f"Bogie/Wheel count: {bogie_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Bogie counter", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
