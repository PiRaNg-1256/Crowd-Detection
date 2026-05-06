import cv2
import json
import os
import winsound

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        threshold = int(data.get("threshold", 10))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        threshold = 10
    print(f"Threshold set to: {threshold}")
    return threshold

def main():
    threshold = load_config()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam (device 0)")
        return

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break

        # Resize to 640px width for performance
        h, w = frame.shape[:2]
        scale = 640 / w
        small = cv2.resize(frame, (640, int(h * scale)))

        # HOG detection on resized frame
        detected = hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
        boxes = detected[0] if len(detected) == 2 else []
        count = len(boxes)

        # Draw green bounding boxes (scaled back to original frame coords)
        for (x, y, bw, bh) in boxes:
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + bw) / scale)
            y2 = int((y + bh) / scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # People count overlay - top-left
        cv2.putText(frame, f"People: {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Crowd Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
