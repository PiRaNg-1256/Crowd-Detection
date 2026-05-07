import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import deque
import statistics


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


# --- NMS tests ---

def test_nms_empty():
    result = nms(np.array([]))
    assert len(result) == 0, f"Expected 0, got {len(result)}"

def test_nms_removes_heavily_overlapping():
    boxes = np.array([[10, 10, 50, 80], [12, 11, 50, 80]])
    result = nms(boxes, overlap_thresh=0.65)
    assert len(result) == 1, f"Expected 1, got {len(result)}"

def test_nms_keeps_separate_people():
    boxes = np.array([[10, 10, 40, 80], [300, 10, 40, 80]])
    result = nms(boxes, overlap_thresh=0.65)
    assert len(result) == 2, f"Expected 2, got {len(result)}"

def test_nms_single_box():
    boxes = np.array([[10, 10, 50, 80]])
    result = nms(boxes, overlap_thresh=0.65)
    assert len(result) == 1, f"Expected 1, got {len(result)}"

def test_nms_returns_int_dtype():
    boxes = np.array([[10, 10, 50, 80], [12, 11, 50, 80]])
    result = nms(boxes)
    assert result.dtype in (np.int32, np.int64), f"Expected int dtype, got {result.dtype}"

# --- Smoothing tests ---

def test_smoothing_median_stable():
    history = deque(maxlen=8)
    for v in [4, 6, 4, 6, 4, 6, 4, 6]:
        history.append(v)
    smoothed = int(statistics.median(history))
    assert smoothed == 5, f"Expected 5, got {smoothed}"

def test_smoothing_single_value():
    history = deque(maxlen=8)
    history.append(3)
    smoothed = int(statistics.median(history))
    assert smoothed == 3, f"Expected 3, got {smoothed}"

def test_smoothing_suppresses_spike():
    history = deque(maxlen=8)
    for v in [4, 4, 4, 4, 4, 4, 4, 20]:  # one big spike
        history.append(v)
    smoothed = int(statistics.median(history))
    assert smoothed == 4, f"Expected 4, got {smoothed}"


if __name__ == "__main__":
    test_nms_empty()
    test_nms_removes_heavily_overlapping()
    test_nms_keeps_separate_people()
    test_nms_single_box()
    test_nms_returns_int_dtype()
    test_smoothing_median_stable()
    test_smoothing_single_value()
    test_smoothing_suppresses_spike()
    print("All tests passed.")
