import cv2
import json
import os
import threading
import numpy as np
from collections import deque
import statistics

try:
    import winsound
    def _beep():
        winsound.Beep(1000, 500)
except ImportError:
    def _beep():
        print("\a", end="", flush=True)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    "threshold": 5,
    "min_confidence": 0.3,
    "smoothing_window": 8,
    "detect_every_n_frames": 3,
    "hog": {"win_stride": [8, 8], "padding": [8, 8], "scale": 1.03},
}

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    config = {**DEFAULTS, **data}
    config["hog"] = {**DEFAULTS["hog"], **data.get("hog", {})}
    try:
        config["threshold"] = int(config["threshold"])
    except (ValueError, TypeError):
        config["threshold"] = DEFAULTS["threshold"]
    if config["threshold"] < 1:
        print("WARNING: threshold must be >= 1, defaulting to 5")
        config["threshold"] = DEFAULTS["threshold"]
    print(f"Threshold set to: {config['threshold']}")
    print(f"Min confidence: {config['min_confidence']}")
    print(f"Smoothing window: {config['smoothing_window']}")
    print(f"Detect every N frames: {config['detect_every_n_frames']}")
    return config

def nms(boxes, overlap_thresh=0.65):
    """Non-maximum suppression. boxes: list/array of (x, y, w, h). Returns filtered array."""
    if len(boxes) == 0:
        return np.array([])
    boxes = np.array(boxes, dtype=float)
    pick = []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype(int)

def main():
    config = load_config()
    threshold = config["threshold"]
    min_conf = config["min_confidence"]
    hog_cfg = config["hog"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam (device 0)")
        return

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    smoothing_window = max(1, int(config.get("smoothing_window", 8)))
    count_history = deque(maxlen=smoothing_window)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    was_overcrowded = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Cannot read frame")
                break

            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                small = cv2.resize(frame, (640, int(h * scale)))
            else:
                scale = 1.0
                small = frame.copy()

            # CLAHE: enhance local contrast for outdoor lighting
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            enhanced_gray = clahe.apply(gray)
            detect_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

            # HOG detection with groupThreshold=1 (merges overlapping rects internally)
            raw = hog.detectMultiScale(
                detect_frame,
                winStride=tuple(hog_cfg["win_stride"]),
                padding=tuple(hog_cfg["padding"]),
                scale=hog_cfg["scale"],
                groupThreshold=1,
            )
            if len(raw) == 2 and len(raw[0]) > 0:
                raw_boxes, raw_weights = raw[0], raw[1].flatten()
                confident = [b for b, wt in zip(raw_boxes, raw_weights) if wt >= min_conf]
                boxes = nms(confident) if len(confident) > 0 else []
            else:
                boxes = []

            count = len(boxes)
            count_history.append(count)
            smoothed_count = int(statistics.median(count_history)) if count_history else 0

            # Draw green bounding boxes on original frame
            for (x, y, bw, bh) in boxes:
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + bw) / scale)
                y2 = int((y + bh) / scale)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"People: {smoothed_count} (raw: {count})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            overcrowded = smoothed_count > threshold

            if overcrowded:
                fh, fw = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (0, 0, 255), 20)
                text = "OVERCROWDED"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
                tx = max(0, (fw - tw) // 2)
                ty = fh // 2 + th // 2
                cv2.putText(frame, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                if not was_overcrowded:
                    threading.Thread(target=_beep, daemon=True).start()

            was_overcrowded = overcrowded
            cv2.imshow("Crowd Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
