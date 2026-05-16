# Crowd Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time webcam crowd detection app that counts people via HOG, shows overlays, and alerts when count exceeds a user-configurable threshold.

**Architecture:** Single `app.py` entry point reads `config.json` at startup, opens webcam via OpenCV, runs HOG detector on every resized frame, draws overlays, and triggers visual+audio alert on threshold crossing. No classes needed — sequential imperative loop.

**Tech Stack:** Python 3.10+, opencv-python, winsound (stdlib)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.json`
- Create: `app.py` (skeleton only — no logic yet)

- [ ] **Step 1: Create `requirements.txt`**

```
opencv-python
```

- [ ] **Step 2: Create `config.json`**

```json
{
  "threshold": 5
}
```

- [ ] **Step 3: Create `app.py` skeleton**

```python
import cv2
import json
import winsound

CONFIG_PATH = "config.json"

def load_config():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify files exist**

Run: `dir D:\crowd-detection` (Windows) or `ls D:\crowd-detection`
Expected: `app.py`, `config.json`, `requirements.txt` all present.

- [ ] **Step 5: Install dependency**

```
pip install opencv-python
```

Expected: `Successfully installed opencv-python-X.X.X` or `already satisfied`.

- [ ] **Step 6: Verify import works**

```
python -c "import cv2; print(cv2.__version__)"
```

Expected: version string printed, no error.

- [ ] **Step 7: Commit**

```
git add app.py config.json requirements.txt
git commit -m "chore: scaffold project with dependencies and config"
```

---

### Task 2: Config Reading + Webcam Open

**Files:**
- Modify: `app.py` — implement `load_config()` and webcam open loop

- [ ] **Step 1: Implement `load_config()`**

Replace the `load_config` stub in `app.py`:

```python
def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        threshold = int(data.get("threshold", 10))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        threshold = 10
    print(f"Threshold set to: {threshold}")
    return threshold
```

- [ ] **Step 2: Verify config reads correctly**

Temporarily add to bottom of `app.py` (remove after test):
```python
if __name__ == "__main__":
    t = load_config()
    assert t == 5, f"Expected 5, got {t}"
    print("Config OK")
```

Run: `python app.py`
Expected output:
```
Threshold set to: 5
Config OK
```

Remove the test lines after confirming.

- [ ] **Step 3: Implement webcam open in `main()`**

```python
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
```

- [ ] **Step 4: Run and verify webcam opens**

```
python app.py
```

Expected: window titled "Crowd Monitor" opens showing live webcam feed. Press Q to close cleanly.

- [ ] **Step 5: Commit**

```
git add app.py
git commit -m "feat: read config and open webcam feed"
```

---

### Task 3: HOG Detection + Overlays

**Files:**
- Modify: `app.py` — add HOG setup and per-frame detection + drawing

- [ ] **Step 1: Initialize HOG detector before the loop**

In `main()`, after `cap = cv2.VideoCapture(0)` and before `while True:`:

```python
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

- [ ] **Step 2: Add detection + overlay inside the loop**

Replace `cv2.imshow("Crowd Monitor", frame)` with:

```python
        # Resize to 640px width for performance
        h, w = frame.shape[:2]
        scale = 640 / w
        small = cv2.resize(frame, (640, int(h * scale)))

        # HOG detection
        boxes, _ = hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
        count = len(boxes)

        # Draw green bounding boxes
        for (x, y, bw, bh) in boxes:
            # Scale coords back to original frame size
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + bw) / scale)
            y2 = int((y + bh) / scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # People count overlay — top-left
        cv2.putText(frame, f"People: {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Crowd Monitor", frame)
```

- [ ] **Step 3: Run and verify overlays**

```
python app.py
```

Expected:
- Green bounding boxes around any visible people
- "People: N" text in top-left corner
- FPS should feel real-time (≥15 FPS on modern laptop)
- Q closes cleanly

- [ ] **Step 4: Commit**

```
git add app.py
git commit -m "feat: HOG people detection with green bounding boxes and count overlay"
```

---

### Task 4: Alert System (Red Border + Beep)

**Files:**
- Modify: `app.py` — add overcrowding visual alert and single-beep transition logic

- [ ] **Step 1: Add overcrowded state tracker before the loop**

In `main()`, after HOG setup and before `while True:`:

```python
    was_overcrowded = False
```

- [ ] **Step 2: Add alert logic inside the loop, after count is computed**

After the `cv2.putText` for the people count and before `cv2.imshow`:

```python
        overcrowded = count > threshold

        if overcrowded:
            # Thick red border around entire frame
            fh, fw = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (0, 0, 255), 20)

            # "OVERCROWDED" text in red, centered-ish
            cv2.putText(frame, "OVERCROWDED", (fw // 2 - 200, fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

            # Beep once on transition into overcrowded
            if not was_overcrowded:
                winsound.Beep(1000, 500)  # 1000 Hz for 500ms

        was_overcrowded = overcrowded
```

- [ ] **Step 3: Run and verify alert**

```
python app.py
```

Manual test steps:
1. With threshold=5 in `config.json`, step into view with several people (or hold up fingers in front of camera to trigger detection).
2. When count > 5: thick red border appears + "OVERCROWDED" text in red + single beep.
3. When count drops ≤ 5: border and text disappear, no beep.
4. Beep plays only once per overcrowded transition (not every frame).

- [ ] **Step 4: Test threshold change**

Edit `config.json`: change `"threshold": 5` to `"threshold": 1`.
Restart: `python app.py`
Expected: alert triggers as soon as 1 person detected.
Restore threshold to 5 after testing.

- [ ] **Step 5: Commit**

```
git add app.py
git commit -m "feat: overcrowding alert with red border, OVERCROWDED text, and single beep"
```

---

### Task 5: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Crowd Detection Camera

Shows your webcam feed and alerts you when too many people are detected.

## Requirements

- Windows PC
- Python 3.10 or newer
- A webcam

## Setup (one-time)

Open Command Prompt and run:

```
pip install opencv-python
```

## How to Run

1. Open Command Prompt
2. Navigate to this folder:
   ```
   cd D:\crowd-detection
   ```
3. Start the program:
   ```
   python app.py
   ```
4. A window opens showing your webcam. Green boxes appear around detected people.
5. Press **Q** to stop.

## Change the Alert Threshold

Open `config.json` with Notepad and change the number:

```json
{ "threshold": 5 }
```

This means: alert when more than 5 people are detected.
Save the file and restart the program for the change to take effect.

## What the Alerts Mean

| What you see | What it means |
|---|---|
| Green boxes + "People: N" | Normal — N people detected |
| Red border + "OVERCROWDED" + beep | Too many people (above threshold) |
```

- [ ] **Step 2: Commit**

```
git add README.md
git commit -m "docs: plain-English README with setup and usage instructions"
```

---

## Final Verification Checklist

Run `python app.py` and confirm all pass:

- [ ] Starts without errors
- [ ] Console prints `Threshold set to: 5`
- [ ] Webcam window opens titled "Crowd Monitor"
- [ ] Green bounding boxes appear around people
- [ ] "People: N" shown top-left every frame
- [ ] When count > threshold: red border + "OVERCROWDED" appears
- [ ] Beep plays once on overcrowded transition, not every frame
- [ ] When count drops ≤ threshold: alert clears
- [ ] Q key exits cleanly (no hanging window or process)
- [ ] Edit `config.json` threshold, restart, alert behavior changes
