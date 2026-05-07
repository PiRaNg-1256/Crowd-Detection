# Crowd Detection Camera

Shows your webcam feed and alerts you when too many people are detected.

---

## Requirements

- Windows PC
- Python 3.10 or newer ([download here](https://www.python.org/downloads/))
- A webcam (built-in laptop camera works fine)

---

## Setup (one-time only)

Open **Command Prompt** and run:

```
pip install opencv-python
```

---

## How to Run

1. Open **Command Prompt**
2. Navigate to the folder where you saved this project, e.g.:
   ```
   cd path\to\crowd-detection
   ```
3. Start the program:
   ```
   python app.py
   ```
4. A window opens showing your webcam feed.
   - Green boxes appear around detected people
   - The count shows in the top-left corner
5. Press **Q** to stop.

---

## Change the Alert Threshold

Open `config.json` with Notepad and change the number:

```json
{
  "threshold": 5
}
```

This means: alert when **more than 5 people** are detected.

Save the file and **restart the program** for the change to take effect.

---

## What the Alerts Mean

| What you see | What it means |
|---|---|
| Green boxes + "People: N" | Normal — N people detected |
| Red border + "OVERCROWDED" + beep | Too many people (count exceeds threshold) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: Cannot open webcam` | Another app is using the camera. Close it and try again. |
| No green boxes appear | Move further back — HOG needs to see full or partial body |
| Program won't start | Make sure you ran `pip install opencv-python` |

---

## Advanced Configuration

All settings live in `config.json`. Open with Notepad to edit. Restart after saving.

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
| `min_confidence` | `0.3` | Filters weak detections (trees, signs). `0.0` = keep all, `1.0` = only very confident. Lower → more detections. Higher → fewer false positives. |
| `smoothing_window` | `8` | Frames to average count over. Higher = more stable, slower to react. |
| `detect_every_n_frames` | `3` | Run detection every N frames. `1` = every frame (slowest). `3` = ~3x faster, boxes update every 150ms. |
| `hog.win_stride` | `[8, 8]` | HOG sliding window step in pixels. Smaller = more thorough, slower. |
| `hog.padding` | `[8, 8]` | Context pixels around each detection window. Larger catches partially-visible people. |
| `hog.scale` | `1.03` | Image pyramid scale step. Smaller (e.g. `1.03`) catches more size variation, slower. |
