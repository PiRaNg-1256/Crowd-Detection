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
2. Navigate to this folder:
   ```
   cd D:\crowd-detection
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
