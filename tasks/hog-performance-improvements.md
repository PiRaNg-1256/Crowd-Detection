# HOG Performance Improvement Plan
## Crowd Detection Camera — Accuracy & Reliability

---

## Root Problems with Vanilla HOG

| Problem | Symptom | Impact |
|---|---|---|
| No NMS | 1 person counted as 3-4 | Count always inflated → false overcrowding alerts |
| No confidence filter | Detects car hoods, signs, trees | False positives → spurious alerts |
| No temporal smoothing | Count fluctuates ±3 every few frames | Alert flickers on/off repeatedly |
| No preprocessing | Outdoor glare/shadow breaks detection | Misses people in bright/dark zones |
| Full-res detection | Slow on large frames | Low FPS, laggy response |

---

## Improvement #1: Non-Maximum Suppression (NMS)
**Impact: HIGH — fixes inflated count**

HOG `detectMultiScale` returns multiple overlapping boxes for same person.
Without NMS, 1 person = 3-4 boxes = count of 4 when threshold is 5 → constant false alert.

**Fix — use OpenCV groupRectangles (zero extra deps):**
```python
# BEFORE (broken — double-counts people)
boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)
count = len(boxes)

# AFTER (correct — merges overlapping boxes)
boxes, weights = hog.detectMultiScale(
    frame,
    winStride=(8, 8),
    padding=(4, 4),
    scale=1.05,
    groupThreshold=1  # built-in grouping that merges overlapping detections
)
count = len(boxes)
```

Or manual NMS using only numpy (also zero extra deps):
```python
def nms(boxes, overlap_thresh=0.65):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(float)
    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2-x1+1) * (y2-y1+1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        overlap = (w*h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype(int)

# Usage: convert (x,y,w,h) → (x1,y1,x2,y2) → NMS → count
rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
filtered = nms(rects)
count = len(filtered)
```

---

## Improvement #2: Confidence Weight Filtering
**Impact: MEDIUM — removes false positives**

HOG returns a confidence weight per detection. Low weight = weak detection = likely false positive (tree, car, sign).

```python
# Filter out weak detections (adjust threshold based on testing)
MIN_CONFIDENCE = 0.3
boxes, weights = hog.detectMultiScale(
    frame, winStride=(8,8), padding=(4,4), scale=1.05, groupThreshold=1
)
# weights is a list of confidence scores
strong_boxes = [b for b, w in zip(boxes, weights.flatten()) if w >= MIN_CONFIDENCE]
count = len(strong_boxes)
```

`MIN_CONFIDENCE` stored in `config.json` so user can tune without code changes.

---

## Improvement #3: Temporal Smoothing
**Impact: HIGH — stops alert flickering**

Count jumps ±2-3 between frames naturally. Without smoothing:
- Threshold = 5, count oscillates between 4 and 6 → alert beeps every 2 seconds

Fix: rolling median over last N frames (median more robust than mean for outliers):

```python
from collections import deque
import statistics

COUNT_HISTORY_SIZE = 8  # from config.json

count_history = deque(maxlen=COUNT_HISTORY_SIZE)

# In main loop:
count_history.append(raw_count)
smoothed_count = int(statistics.median(count_history)) if count_history else 0

# Use smoothed_count for alert logic, display raw_count in overlay
```

---

## Improvement #4: CLAHE Preprocessing
**Impact: MEDIUM — better outdoor lighting**

Outdoor scenes have harsh shadows and glare. CLAHE enhances local contrast, making
people detectable even in dark shadow areas or bright patches.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# In main loop (before detection):
gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
enhanced_gray = clahe.apply(gray)
# Convert back to BGR (HOG works on both, BGR gives slightly more info)
enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

# Run HOG on enhanced_frame instead of raw frame
boxes, weights = hog.detectMultiScale(enhanced_frame, ...)
```

---

## Improvement #5: HOG Parameter Tuning
**Impact: MEDIUM — accuracy vs speed balance**

```python
# Current (original) — balanced
hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)

# Recommended — slightly better accuracy, ~same speed
hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.03)
# Larger padding = more context around each window = catches partially-visible people
# Smaller scale step (1.03 vs 1.05) = more scale levels = catches more size variation
```

Expose `win_stride`, `padding`, `scale` in `config.json` for user experimentation.

---

## Improvement #6: Frame Skipping with Cached Boxes
**Impact: HIGH — FPS improvement**

HOG is the bottleneck. Running every frame at 640px ≈ 20FPS.
Run detection every 3rd frame, display last known boxes on other frames:

```python
DETECT_EVERY = 3  # from config.json

frame_idx = 0
cached_boxes = []
cached_count = 0

while True:
    ret, frame = cap.read()
    frame_resized = resize_frame(frame, width=640)

    if frame_idx % DETECT_EVERY == 0:
        # Run full detection
        cached_boxes, cached_count = detect(frame_resized)
    
    # Always draw cached boxes (fast)
    draw_overlays(frame_resized, cached_boxes, cached_count)
    frame_idx += 1
```

Result: 3x FPS improvement with minimal accuracy loss (boxes update ~every 150ms).

---

## Updated config.json

```json
{
  "threshold": 10,
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

---

## Combined Impact

| Improvement | Accuracy Gain | FPS Impact |
|---|---|---|
| NMS (groupThreshold) | +++ removes duplicates | Neutral |
| Confidence filter | ++ removes false positives | Neutral |
| Temporal smoothing | +++ stops flickering | Neutral |
| CLAHE preprocessing | ++ outdoor lighting | -15% FPS |
| Param tuning | + catches more sizes | -10% FPS |
| Frame skipping | - (cache lag) | +200% FPS |

Net result: significantly more accurate count, stable alerts, ~2x FPS.

---

## Files to Modify

| File | Changes |
|---|---|
| `app.py` | Add NMS, confidence filter, temporal smoothing, CLAHE, frame skipping |
| `config.json` | Add `min_confidence`, `smoothing_window`, `detect_every_n_frames`, `hog` block |
| `requirements.txt` | Add `numpy` (needed for NMS — likely already installed with opencv) |
| `README.md` | Document new config options |
