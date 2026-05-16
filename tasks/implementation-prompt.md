# Implementation Prompt — Crowd Detection Camera

Paste this entire prompt at the start of a new Claude Code session inside the `crowd-detection/` folder.

---

## PROMPT (copy everything below this line)

---

You are building a **real-time crowd detection system** for a non-developer user. The full requirements are in `D:\crowd-detection\tasks\prd-crowd-detection-camera.md`. Read it before doing anything else.

## Active Mode

Use **caveman mode** for all responses to minimize token usage:
- Invoke skill: `caveman:caveman`
- Drop filler words, use fragments, keep technical substance intact
- Code blocks stay normal — only prose gets compressed

## Workflow — Follow This Order Exactly

### Phase 1 — Understand Before Building
1. Invoke skill: `superpowers:brainstorming` — confirm you understand the full scope before writing any code
2. Read `tasks/prd-crowd-detection-camera.md` completely
3. Run skill: `gsd-spec-phase` — produce a concise spec confirming tech stack, file structure, and acceptance criteria mapping
4. Ask the user: **"What number should trigger the overcrowding alert?"** — set this as `threshold` in `config.json`

### Phase 2 — Plan
5. Run skill: `gsd-plan-phase` — break work into milestones:
   - Milestone 1: Project scaffolding (requirements.txt, config.json, README)
   - Milestone 2: Webcam feed + HOG person detection + overlay
   - Milestone 3: Alert system (visual border + beep)
   - Milestone 4: Polish + manual start/stop
6. Run skill: `superpowers:writing-plans` — write granular task list before touching any code

### Phase 3 — Build (milestone by milestone)
7. Run skill: `gsd-execute-phase` for each milestone
8. Run skill: `superpowers:test-driven-development` — for each functional requirement, verify it works before moving to next
9. After each milestone, run skill: `superpowers:verification-before-completion` — do NOT proceed until current milestone passes

### Phase 4 — Debug (if needed)
10. If anything breaks, run skill: `superpowers:systematic-debugging` — do NOT guess at fixes; find root cause first

### Phase 5 — Finish
11. Run skill: `gsd-validate-phase` — test the full happy path end-to-end
12. Run skill: `superpowers:verification-before-completion` — final check before declaring done
13. Run skill: `gsd-ship` — clean up, write README with run instructions for non-developer

## Tech Stack (do not deviate)

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | Simplest for this use case |
| Webcam + display + detection | `opencv-python` | HOG detector built-in, no extra deps |
| Audio alert | `winsound` (built-in) | No extra install, Windows native |
| Config | `config.json` | Non-developer can edit without code |

**Install command:**
```
pip install opencv-python
```

**Detection code (use exactly this):**
```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)
count = len(boxes)
```

## File Structure to Create

```
crowd-detection/
├── app.py            # Main program entry point
├── config.json       # { "threshold": <user_number> }
├── requirements.txt  # opencv-python only
└── README.md         # Plain-English run instructions
```

## Key Behaviours (from PRD)

- Use OpenCV HOG detector — NO AI models, NO internet required
- Draw **green bounding boxes** around each detected person
- Show **"People: N"** count in top-left corner every frame
- When count > threshold → **thick red border** + **"OVERCROWDED"** text in red
- Beep **once** on transition into overcrowded state (not every frame)
- Press **Q** to quit cleanly (release camera, close window)
- Resize frame to 640px width before detection for performance

## Context Minimization Rules

- Do NOT look up or import `ultralytics`, `torch`, or any AI library
- Do NOT generate more than one file at a time
- Do NOT add features not in the PRD (no recording, no logging, no remote access)
- After each milestone completes, summarize in ≤3 sentences before moving on
- If unsure about a decision, pick the simpler option and note it

## Definition of Done

The project is complete when:
- [ ] `python app.py` starts without errors
- [ ] Live webcam feed shows with person count overlay
- [ ] Changing `threshold` in `config.json` and restarting changes alert behavior
- [ ] Red border + "OVERCROWDED" appears when count exceeds threshold
- [ ] Beep plays once when overcrowding is first detected
- [ ] `Q` key exits cleanly
- [ ] README explains how to install and run in plain English (no coding knowledge assumed)

---

## END OF PROMPT
