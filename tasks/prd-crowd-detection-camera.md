# PRD: Crowd Detection Camera System

## Introduction

A real-time crowd monitoring system using a laptop webcam that detects how many people are visible, compares against a user-defined threshold, and alerts immediately when overcrowding is detected. Designed for outdoor use with manual start/stop control.

---

## Goals

- Detect and count people visible through the laptop webcam in real time
- Allow user to set a custom overcrowding threshold (number of people)
- Display live video with visual overlays (bounding boxes, count)
- Trigger on-screen visual alert + audible beep when threshold is exceeded
- Start and stop detection manually via a button/key press
- Require no coding knowledge to run after setup

---

## How It Works (Plain English)

1. User runs the program
2. A window opens showing the live laptop camera feed
3. OpenCV's built-in HOG people detector scans each video frame and draws boxes around every person it sees
4. A counter in the corner shows how many people are detected right now
5. User sets their threshold (e.g. "flag if more than 8 people")
6. When count exceeds threshold → screen border turns red + beep sounds
7. When count drops back below threshold → alert clears automatically
8. User presses `Q` or clicks Stop to end the session

---

## User Stories

### US-001: Configure overcrowding threshold
**Description:** As a user, I want to set my own number so that the system flags overcrowding at my chosen level.

**Acceptance Criteria:**
- [ ] `config.json` file exists with a `threshold` field (default: 10)
- [ ] User can change the number in `config.json` without touching any code
- [ ] Program reads threshold at startup and prints it to console
- [ ] Changing threshold takes effect on next program start

### US-002: Live webcam feed with person detection overlay
**Description:** As a user, I want to see the camera feed with boxes drawn around people so I know the system is working.

**Acceptance Criteria:**
- [ ] Window opens showing live webcam feed at reasonable FPS (≥15)
- [ ] Green bounding box drawn around each detected person
- [ ] Person count displayed in top-left corner as "People: N"
- [ ] Overlay updates every frame in real time

### US-003: Overcrowding visual alert
**Description:** As a user, I want a clear visual warning when too many people are detected so I can react immediately.

**Acceptance Criteria:**
- [ ] When count > threshold: screen border turns red + "OVERCROWDED" text shown in large red font
- [ ] When count ≤ threshold: border returns to normal, warning text disappears
- [ ] Alert activates within 1 second of threshold being crossed

### US-004: Audible beep alert
**Description:** As a user, I want to hear a beep when overcrowding is detected so I don't have to watch the screen constantly.

**Acceptance Criteria:**
- [ ] Beep plays when overcrowding is first detected
- [ ] Beep does NOT repeat every frame (only on transition from normal → overcrowded)
- [ ] Works on Windows without external audio software

### US-005: Manual start/stop
**Description:** As a user, I want to start and stop the system myself so I control when monitoring is active.

**Acceptance Criteria:**
- [ ] Program starts when user runs `python app.py`
- [ ] Pressing `Q` key in the video window stops the program cleanly
- [ ] Camera is released and window closes on exit (no hanging processes)

---

## Functional Requirements

- **FR-1:** Read `threshold` from `config.json` at startup; default to 10 if file missing
- **FR-2:** Open laptop webcam (device index 0) using OpenCV
- **FR-3:** Use OpenCV HOG people detector (`cv2.HOGDescriptor` with default people detector SVM) on each frame to detect persons
- **FR-4:** Draw green bounding boxes around each detected person
- **FR-5:** Display live person count as overlay text on video frame
- **FR-6:** When count > threshold: draw thick red border around frame + show "OVERCROWDED" text
- **FR-7:** Play a single beep (Windows `winsound`) only on the first frame where count crosses threshold
- **FR-8:** Display feed in an OpenCV window titled "Crowd Monitor"
- **FR-9:** Exit cleanly when `Q` key is pressed

---

## Non-Goals (Out of Scope)

- No face recognition or identification of individuals
- No saving video recordings
- No remote/network monitoring
- No mobile app or web dashboard
- No multi-camera support (single laptop webcam only)
- No scheduling (start/stop is manual only)
- No cloud integration or data upload
- No historical logs or reports

---

## Technical Considerations

**Language:** Python 3.10+

**Key Libraries:**
| Library | Purpose |
|---|---|
| `opencv-python` | Webcam capture, HOG detection, display, drawing overlays |
| `winsound` (built-in) | Beep alert (Windows only) |

**Why HOG (Histogram of Oriented Gradients)?**
- Built into OpenCV — zero extra installs beyond `opencv-python`
- No AI model download, no internet needed
- Runs entirely on CPU, fast enough for real-time detection
- Trade-off: less accurate than YOLO when people overlap, but sufficient for general crowd counting

**HOG Detection Call:**
```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)
```

**File Structure:**
```
crowd-detection/
├── app.py           # Main program
├── config.json      # User settings (threshold)
├── requirements.txt # pip dependencies (opencv-python only)
└── README.md        # How to run
```

**Performance Notes:**
- HOG runs at ~15-25 FPS on modern laptop CPU at 640px width
- Frame resized to 640px width before detection for speed
- Detection runs every frame

---

## Success Metrics

- System detects people accurately enough for user to trust alerts
- Alert triggers within 1 second of threshold being crossed
- Program starts with a single command: `python app.py`
- Non-developer can change the threshold by editing one number in `config.json`
- No crashes during a 30-minute continuous run

---

## Open Questions

- What specific threshold number does the user want as default? (Currently set to 10 in config)
- Should future version support saving alert screenshots automatically?
- Should beep frequency/duration be configurable?
