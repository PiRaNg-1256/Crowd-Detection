# Improvement Prompt — HOG Performance Upgrade

Paste this at the start of a new Claude Code session inside the `crowd-detection/` folder.
Use AFTER the base `app.py` is already working.

---

## PROMPT (copy everything below this line)

---

You are upgrading an existing `app.py` crowd detection system to be significantly more
accurate. Read `tasks/hog-performance-improvements.md` completely before touching any code.
Then read the existing `app.py` and `config.json` to understand current state.

## Active Mode

Invoke skill `caveman:caveman` — compress all prose responses to save tokens.
Code blocks stay normal.

## Pre-Flight Checks (do these first, in order)

1. Read `app.py` — understand current HOG detection flow end-to-end
2. Read `config.json` — note current keys
3. Read `tasks/hog-performance-improvements.md` — understand all 6 improvements
4. Run `python app.py` — confirm baseline works before any changes

## Workflow

### Phase 1 — Plan
Run skill: `superpowers:writing-plans`

Break into 4 atomic milestones:
- **M1:** NMS + confidence filtering (accuracy — no new deps)
- **M2:** Temporal smoothing (stability — no new deps)
- **M3:** CLAHE preprocessing (outdoor lighting — no new deps)
- **M4:** Frame skipping + config.json expansion + README update

Each milestone = one set of changes to `app.py` + verify it still runs.

### Phase 2 — Build (one milestone at a time)

For each milestone:
1. Run skill: `superpowers:verification-before-completion` — test current state passes first
2. Make the changes
3. Run `python app.py` — verify it launches, camera opens, detection visible
4. Run skill: `superpowers:verification-before-completion` — confirm milestone done
5. ONLY then move to next milestone

**Never combine milestones.** One change set, one test, one confirm.

### Phase 3 — Debug (if needed)

If anything breaks: run skill `superpowers:systematic-debugging`
- Reproduce the exact error
- Find root cause before changing anything
- One fix, re-test

### Phase 4 — Finish

Run skill: `gsd-validate-phase` — test full happy path:
- Camera opens ✓
- Bounding boxes appear ✓
- Count shown ✓
- Alert triggers at threshold ✓
- Beep plays once ✓
- Q exits cleanly ✓

Run skill: `gsd-ship` — update README with new config options.

## Improvements to Implement (in order)

### M1: NMS + Confidence Filtering

**Goal:** Each real person = exactly 1 bounding box. Fake detections (trees, signs) filtered out.

Replace the detection call in `app.py` with:
```python
import numpy as np
from collections import deque
import statistics

def nms(boxes, overlap_thresh=0.65):
    if len(boxes) == 0:
        return np.array([])
    boxes = np.array(boxes, dtype=float)
    pick = []
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
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

# In detection:
boxes, weights = hog.detectMultiScale(
    frame_resized,
    winStride=tuple(config.get("hog", {}).get("win_stride", [8, 8])),
    padding=tuple(config.get("hog", {}).get("padding", [8, 8])),
    scale=config.get("hog", {}).get("scale", 1.03),
    groupThreshold=1
)
if len(boxes) > 0 and len(weights) > 0:
    min_conf = config.get("min_confidence", 0.3)
    confident_boxes = [b for b, w in zip(boxes, weights.flatten()) if w >= min_conf]
    if len(confident_boxes) > 0:
        boxes = nms(confident_boxes)
    else:
        boxes = []
count = len(boxes)
```

### M2: Temporal Smoothing

**Goal:** Alert never flickers. Count stable across frames.

Add after detection in the main loop:
```python
# Initialize before loop:
smoothing_window = config.get("smoothing_window", 8)
count_history = deque(maxlen=smoothing_window)

# In loop, after getting raw count:
count_history.append(count)
smoothed_count = int(statistics.median(count_history)) if count_history else 0

# Use smoothed_count for alert logic
# Display both: "People: N (raw: M)" optional
```

### M3: CLAHE Preprocessing

**Goal:** Detect people in shadows and bright outdoor patches.

Add before detection:
```python
# Initialize once before loop:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# In loop, before detection:
gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
enhanced = clahe.apply(gray)
detect_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Run HOG on detect_frame instead of frame_resized
boxes, weights = hog.detectMultiScale(detect_frame, ...)
# Draw overlays on original frame_resized (not detect_frame)
```

### M4: Frame Skipping + Config Expansion

**Goal:** 2-3x FPS improvement.

```python
# Initialize before loop:
detect_every = config.get("detect_every_n_frames", 3)
frame_idx = 0
cached_boxes = []
cached_smoothed_count = 0

# In loop:
if frame_idx % detect_every == 0:
    # Run full detection pipeline (NMS + smoothing)
    cached_boxes, cached_smoothed_count = run_detection(frame_resized)
frame_idx += 1

# Always draw cached_boxes
draw_overlays(frame_resized, cached_boxes, cached_smoothed_count)
```

Update `config.json` to add:
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

## Requirements: Do NOT add

- No new pip installs (`numpy` is bundled with opencv-python already)
- No AI/ML models
- No features not in this prompt
- No changes to alert/beep logic (that already works)

## Definition of Done

- [ ] One person detected = one box (not 3-4)
- [ ] Fake detections (signs, cars, walls) reduced significantly
- [ ] Count doesn't jump ±3 between frames when scene is static
- [ ] `config.json` has all 4 new tuneable fields
- [ ] Camera still opens and runs at ≥15 FPS
- [ ] Q still exits cleanly
- [ ] README documents what each new config field does

---

## END OF PROMPT
