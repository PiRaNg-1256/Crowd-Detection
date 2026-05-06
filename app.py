import cv2
import json
import winsound

CONFIG_PATH = "config.json"

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break

        cv2.imshow("Crowd Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
