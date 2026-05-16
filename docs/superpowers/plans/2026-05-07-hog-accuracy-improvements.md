# HOG Accuracy Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade existing HOG crowd detection with NMS, confidence filtering, temporal smoothing, CLAHE preprocessing, and frame skipping to eliminate false detections, stop alert flickering, and improve FPS.

**Architecture:** All changes in `app.py`. `load_config()` refactored to return full config dict (not just threshold). NMS is a pure function. Detection pipeline: resize → CLAHE → HOG → confidence filter → NMS → median smooth → alert. Frame skipping caches last detection result and reuses on non-detection frames.

**Tech Stack:** Python 3.10+, opencv-python, numpy (bundled with opencv), collections.deque, statistics (stdlib)

---

### Task 1 (M1): NMS + Confidence Filtering

**Files:**
- Modify: `D:\crowd-detection\app.py`
- Create: `D:\crowd-detection\tests\test_nms.py`

**What changes:**
- `load_config()` refactored to return full config dict with defaults
- `nms()` pure function added at module level
- Detection call updated: `groupThreshold=1`, confidence filter, then NMS
- HOG params read from config `hog` block

- [ ] **Step 1: Write NMS unit tests**

Create `D:\crowd-detection\tests\test_nms.py`:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import nms directly — copy the function here so tests are self-contained
def nms(boxes, overlap_thresh=0.65):
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


def test_nms_empty():
    result = nms(np.array([]))
    assert len(result) == 0

def test_nms_removes_heavily_overlapping():
    # Same person detected twice — almost identical boxes
    boxes = np.array([[10, 10, 50, 80], [12, 11, 50, 80]])
    result = nms(boxes, overlap_thresh=0.65)
    assert len(result) == 1

def test_nms_keeps_separate_people():
    # Two people far apart — no overlap
    boxes = np.array([[10, 10, 40, 80], [300, 10, 40, 80]])
    result = nms(boxes, overlap_thresh=0.65)
    assert len(result) == 2

def test_nms_single_box():
    boxes = np.array([[10, 10, 50, 80]])
    result = nms(boxes, overlap_thresh=0.65)
    assert len(result) == 1

def test_nms_returns_int_dtype():
    boxes = np.array([[10, 10, 50, 80], [12, 11, 50, 80]])
    result = nms(boxes)
    assert result.dtype in (np.int32, np.int64, int)

if __name__ == "__main__":
    test_nms_empty()
    test_nms_removes_heavily_overlapping()
    test_nms_keeps_separate_people()
    test_nms_single_box()
    test_nms_returns_int_dtype()
    print("All NMS tests passed.")
```

- [ ] **Step 2: Run tests to verify they pass (NMS logic is standalone)**

```
python D:\crowd-detection\tests\test_nms.py
```

Expected output: `All NMS tests passed.`

- [ ] **Step 3: Rewrite app.py with NMS + confidence filter + refactored load_config**

Write the complete `D:\crowd-detection\app.py`:

```python
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
    # hog sub-dict: merge, don't replace
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
    """Non-maximum suppression. boxes: list/array of (x, y, w, h). Returns filtered (x,y,w,h) array."""
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

            # HOG detection with groupThreshold=1 (built-in overlap merging)
            raw = hog.detectMultiScale(
                small,
                winStride=tuple(hog_cfg["win_stride"]),
                padding=tuple(hog_cfg["padding"]),
                scale=hog_cfg["scale"],
                groupThreshold=1,
            )
            if len(raw) == 2 and len(raw[0]) > 0:
                raw_boxes, raw_weights = raw[0], raw[1].flatten()
                # Confidence filter
                confident = [b for b, wt in zip(raw_boxes, raw_weights) if wt >= min_conf]
                # NMS
                boxes = nms(confident) if len(confident) > 0 else []
            else:
                boxes = []

            count = len(boxes)

            # Draw green bounding boxes on original frame
            for (x, y, bw, bh) in boxes:
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + bw) / scale)
                y2 = int((y + bh) / scale)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"People: {count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            overcrowded = count > threshold

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
```

- [ ] **Step 4: Verify syntax + imports**

```
python -c "import ast; ast.parse(open('D:/crowd-detection/app.py').read()); print('syntax OK')"
python -c "import cv2, numpy, json, os, threading, collections, statistics; print('imports OK')"
```

Expected: both print OK.

- [ ] **Step 5: Run NMS tests against app.py's nms function**

```
python D:\crowd-detection\tests\test_nms.py
```

Expected: `All NMS tests passed.`

- [ ] **Step 6: Commit**

```
git -C D:\crowd-detection add app.py tests/test_nms.py
git -C D:\crowd-detection commit -m "feat: NMS + confidence filtering + refactor load_config to return full dict"
git -C D:\crowd-detection push
```

---

### Task 2 (M2): Temporal Smoothing

**Files:**
- Modify: `D:\crowd-detection\app.py` — add deque median smoothing to detection loop

**What changes:**
- `count_history = deque(maxlen=smoothing_window)` initialized before loop
- Raw `count` appended each detection frame
- `smoothed_count` = median of history, used for alert logic
- Overlay shows both: `"People: N"` (smoothed) with raw in parentheses

- [ ] **Step 1: Verify M1 state**

```
python -c "import ast; ast.parse(open('D:/crowd-detection/app.py').read()); print('M1 syntax OK')"
python D:\crowd-detection\tests\test_nms.py
```

Both must pass before proceeding.

- [ ] **Step 2: Write smoothing unit test**

Append to `D:\crowd-detection\tests\test_nms.py` (or create separate file — add to test_nms.py):

```python
from collections import deque
import statistics

def test_smoothing_median_stable():
    history = deque(maxlen=8)
    # Simulate oscillating raw counts: 4,6,4,6,4,6,4,6
    for v in [4, 6, 4, 6, 4, 6, 4, 6]:
        history.append(v)
    smoothed = int(statistics.median(history))
    assert smoothed == 5  # median of [4,4,4,4,6,6,6,6] = 5

def test_smoothing_single_value():
    history = deque(maxlen=8)
    history.append(3)
    smoothed = int(statistics.median(history))
    assert smoothed == 3

if __name__ == "__main__":
    test_nms_empty()
    test_nms_removes_heavily_overlapping()
    test_nms_keeps_separate_people()
    test_nms_single_box()
    test_nms_returns_int_dtype()
    test_smoothing_median_stable()
    test_smoothing_single_value()
    print("All tests passed.")
```

Run: `python D:\crowd-detection\tests\test_nms.py`
Expected: `All tests passed.`

- [ ] **Step 3: Add temporal smoothing to app.py**

In `main()`, after `hog.setSVMDetector(...)` and before the loop, add:

```python
    smoothing_window = config.get("smoothing_window", 8)
    count_history = deque(maxlen=smoothing_window)
```

Inside the loop, after `count = len(boxes)`, add:

```python
            count_history.append(count)
            smoothed_count = int(statistics.median(count_history)) if count_history else 0
```

Replace all uses of `count` in alert logic and overlay with `smoothed_count`:

```python
            # Overlay: show smoothed count, raw in parentheses
            cv2.putText(frame, f"People: {smoothed_count} (raw: {count})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            overcrowded = smoothed_count > threshold
```

- [ ] **Step 4: Verify syntax**

```
python -c "import ast; ast.parse(open('D:/crowd-detection/app.py').read()); print('M2 syntax OK')"
```

- [ ] **Step 5: Run all tests**

```
python D:\crowd-detection\tests\test_nms.py
```

Expected: `All tests passed.`

- [ ] **Step 6: Commit**

```
git -C D:\crowd-detection add app.py tests/test_nms.py
git -C D:\crowd-detection commit -m "feat: temporal smoothing via rolling median (smoothing_window config key)"
git -C D:\crowd-detection push
```

---

### Task 3 (M3): CLAHE Preprocessing

**Files:**
- Modify: `D:\crowd-detection\app.py` — add CLAHE before HOG detection

**What changes:**
- `clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` initialized before loop
- Each frame: convert to gray → CLAHE → back to BGR → feed to HOG
- Original colour frame still used for drawing overlays

- [ ] **Step 1: Verify M2 state**

```
python -c "import ast; ast.parse(open('D:/crowd-detection/app.py').read()); print('M2 syntax OK')"
python D:\crowd-detection\tests\test_nms.py
```

Both must pass.

- [ ] **Step 2: Add CLAHE init before loop**

In `main()`, after `hog.setSVMDetector(...)` and before `count_history = deque(...)`, add:

```python
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

- [ ] **Step 3: Add CLAHE preprocessing inside loop**

After the resize block (`small = ...`) and before the HOG detection call, add:

```python
            # CLAHE: enhance local contrast for outdoor lighting
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            enhanced_gray = clahe.apply(gray)
            detect_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
```

Change the HOG call to use `detect_frame` instead of `small`:

```python
            raw = hog.detectMultiScale(
                detect_frame,
                winStride=tuple(hog_cfg["win_stride"]),
                padding=tuple(hog_cfg["padding"]),
                scale=hog_cfg["scale"],
                groupThreshold=1,
            )
```

Keep all bounding box drawing on `frame` (original colours) — no change needed there.

- [ ] **Step 4: Verify syntax**

```
python -c "import ast; ast.parse(open('D:/crowd-detection/app.py').read()); print('M3 syntax OK')"
```

- [ ] **Step 5: Verify CLAHE API available**

```
python -c "import cv2; c=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); print('CLAHE OK')"
```

Expected: `CLAHE OK`

- [ ] **Step 6: Commit**

```
git -C D:\crowd-detection add app.py
git -C D:\crowd-detection commit -m "feat: CLAHE preprocessing for outdoor lighting robustness"
git -C D:\crowd-detection push
```

---

### Task 4 (M4): Frame Skipping + Config Expansion + README

**Files:**
- Modify: `D:\crowd-detection\app.py` — frame skipping loop
- Modify: `D:\crowd-detection\config.json` — add all new keys
- Modify: `D:\crowd-detection\README.md` — document new config options

**What changes:**
- Detection runs only every `detect_every_n_frames` frames
- Cached boxes + smoothed_count reused on skipped frames
- `config.json` gets all 4 new tuneable fields
- README gets new "Advanced Configuration" section

- [ ] **Step 1: Verify M3 state**

```
python -c "import ast; ast.parse(open('D:/crowd-detection/app.py').read()); print('M3 syntax OK')"
python D:\crowd-detection\tests\test_nms.py
```

Both must pass.

- [ ] **Step 2: Add frame-skip variables before loop**

In `main()`, after `count_history = deque(maxlen=smoothing_window)`, add:

```python
    detect_every = max(1, int(config.get("detect_every_n_frames", 3)))
    frame_idx = 0
    cached_boxes = []
    cached_smoothed_count = 0
```

- [ ] **Step 3: Wrap detection block in frame-skip condition**

The full detection pipeline (resize → CLAHE → HOG → confidence filter → NMS → smooth) should only run when `frame_idx % detect_every == 0`. On other frames, reuse cached values.

Replace the detection section inside the loop with:

```python
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                small = cv2.resize(frame, (640, int(h * scale)))
            else:
                scale = 1.0
                small = frame.copy()

            if frame_idx % detect_every == 0:
                # CLAHE preprocessing
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                enhanced_gray = clahe.apply(gray)
                detect_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

                # HOG detection
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

                cached_boxes = boxes
                cached_smoothed_count = smoothed_count
            else:
                # Reuse last detection
                boxes = cached_boxes
                smoothed_count = cached_smoothed_count

            frame_idx += 1
```

- [ ] **Step 4: Update config.json**

Write `D:\crowd-detection\config.json`:

```json
{
  "threshold": 5,
  "min_confidence": 0.3,
  "smoothing_window": 8,
  "detect_every_n_frames": 3,
  "hog": {
    "win_stride": [8, 8],
    "padding": [8, 8],
    "scale": 1.03
  }
}
```

- [ ] **Step 5: Verify syntax + config loads**

```
python -c "import ast; ast.parse(open('D:/crowd-detection/app.py').read()); print('M4 syntax OK')"
python -c "import json; c=json.load(open('D:/crowd-detection/config.json')); print('threshold:', c['threshold']); print('hog scale:', c['hog']['scale']); print('config OK')"
```

Expected:
```
M4 syntax OK
threshold: 5
hog scale: 1.03
config OK
```

- [ ] **Step 6: Add Advanced Configuration section to README.md**

Append to the end of `D:\crowd-detection\README.md`:

```markdown
---

## Advanced Configuration

All settings live in `config.json`. Open with Notepad to edit. Restart the program after saving.

```json
{
  "threshold": 5,
  "min_confidence": 0.3,
  "smoothing_window": 8,
  "detect_every_n_frames": 3,
  "hog": {
    "win_stride": [8, 8],
    "padding": [8, 8],
    "scale": 1.03
  }
}
```

| Field | Default | What it does |
|---|---|---|
| `threshold` | `5` | Alert triggers when people count exceeds this number |
| `min_confidence` | `0.3` | Filter out weak detections (0.0 = keep all, 1.0 = keep only very confident). Lower → more detections. Higher → fewer false positives. |
| `smoothing_window` | `8` | Number of past frames to average count over. Higher = more stable but slower to react. |
| `detect_every_n_frames` | `3` | Run detection every N frames. `1` = every frame (slowest, most responsive). `3` = 3x faster, boxes update every ~150ms. |
| `hog.win_stride` | `[8, 8]` | HOG sliding window step size in pixels. Smaller = more thorough but slower. |
| `hog.padding` | `[8, 8]` | Pixels of context around each detection window. Larger catches partially-visible people. |
| `hog.scale` | `1.03` | Image pyramid scale factor. Smaller (e.g. 1.03) = more size levels = finds more people = slower. |
```

- [ ] **Step 7: Run all tests**

```
python D:\crowd-detection\tests\test_nms.py
```

Expected: `All tests passed.`

- [ ] **Step 8: Commit**

```
git -C D:\crowd-detection add app.py config.json README.md
git -C D:\crowd-detection commit -m "feat: frame skipping, config.json expanded with all tuneable fields, README updated"
git -C D:\crowd-detection push
```

---

## Final Verification Checklist

Run `python D:\crowd-detection\tests\test_nms.py` — all pass.

Run `python D:\crowd-detection\app.py` manually and confirm:

- [ ] Console prints threshold, min_confidence, smoothing_window, detect_every_n_frames
- [ ] Window opens titled "Crowd Monitor"
- [ ] "People: N (raw: M)" shown top-left
- [ ] One box per person (not 3-4)
- [ ] Count stable when scene is static (median smoothing working)
- [ ] When count > threshold: red border + OVERCROWDED + single beep
- [ ] Q exits cleanly
- [ ] Edit `config.json` threshold, restart → alert changes
