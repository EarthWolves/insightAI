# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo structure

Two independent product MVPs, each self-contained in its own folder:

```
insightAI/
├── construction_site/   — Construction site intelligence (InsightAI)
│   ├── detect.py        — Grounding DINO core (shared pattern)
│   ├── camera.py        — Video/camera pipeline with background inference thread
│   ├── dashboard.py     — Main 1600×900 OpenCV dashboard
│   ├── requirements.txt
│   └── huntington-apartments-time-lapse-ytshorts.savetube.vip.mp4
└── recycling_line/      — Recycling conveyor intelligence (RecycleVision)
    ├── detect.py        — Grounding DINO core (standalone copy)
    ├── conveyor.py      — Video pipeline (adapted from camera.py)
    ├── dashboard.py     — Main 1600×900 OpenCV dashboard
    ├── requirements.txt
    └── conveyor.mp4                 ← sample video
```

## Setup

```bash
pip install -r construction_site/requirements.txt
# (recycling_line/requirements.txt is identical)
```

---

## construction_site — InsightAI

Tracks construction equipment (excavator, bulldozer, crane, dump truck, worker) on a timelapse video and overlays a project intelligence dashboard.

**Run:**
```bash
cd construction_site
python dashboard.py --video huntington-apartments-time-lapse-ytshorts.savetube.vip.mp4
python dashboard.py --video huntington-apartments-time-lapse-ytshorts.savetube.vip.mp4 --speed 0.25 --save-video

# Standalone detection (no dashboard):
python camera.py --video timelapse.mp4 --classes "excavator, crane, worker"
python detect.py --image frame.jpg --classes "crane, worker"
```

**Dashboard panels (right column):** Phase Progress (7 phases) → Real-Time Events (15 frame-triggered) → Material Inventory → AI Optimization insights

**Demo context:** Huntington Apartments Phase II, Apex Construction LLC, $18.2M, Oct 2026 target

---

## recycling_line — RecycleVision

Counts objects travelling on a conveyor belt using background subtraction + centroid tracking. No ML model required — runs in real-time on any hardware.

**Run:**
```bash
cd recycling_line
python dashboard.py --video conveyor.mp4
python dashboard.py --video conveyor.mp4 --speed 0.5 --save-video
# Restrict detection to just the belt area (avoids machinery edges):
python dashboard.py --video conveyor.mp4 --roi 80,60,920,500
# Adjust counting line position (0.0–1.0 fraction of frame width):
python dashboard.py --video conveyor.mp4 --line 0.6
# Tune sensitivity: lower threshold = more sensitive to small/dim items
python dashboard.py --video conveyor.mp4 --threshold 25 --min-area 500

# Adjust line position while running: + / - keys (10px per press)
```

**Files:**
- `counter.py` — `ConveyorDetector` (MOG2), `CentroidTracker` (centroid assignment), `CountState` (crossings, rate, events)
- `dashboard.py` — counting UI (no external imports except counter.py + OpenCV/numpy)
- `conveyor.py` + `detect.py` — legacy Grounding DINO pipeline (still works if needed)

**How the counter works:**
1. MOG2 background subtractor learns the static belt → foreground binary mask
2. Morphological open/close/dilate cleans up blobs → `cv2.findContours` → bounding boxes
3. `CentroidTracker` nearest-centroid assignment → stable IDs across frames
4. Virtual counting line: when `prev_x < line_x <= curr_x` for a tracked object → count += 1
5. First `warmup_frames=60` frames excluded (MOG2 still learning background)

**Tuning for new footage:**
- `--roi` is most important — exclude belt frame/machinery from detection zone
- `--threshold` (MOG2 `varThreshold`): lower → more sensitive; raise if too many false positives
- `--min-area`: raise to ignore small noise blobs; lower if items are small
- `--line` (0.0–1.0): place counting line in the clear section of belt, not near edges
- `+`/`-` keys adjust line by 10px while running

**Dashboard layout:** Large video (right, 1176×661, 16:9), left stats panel (400px):
- ITEMS COUNTED — giant hero number with pulse glow on crossing
- THROUGHPUT — items/min (30-second rolling window) + items/hr
- SHIFT TARGET — progress toward 5,000 item target
- BELT STATUS — objects on belt right now + load label
- EVENTS — 8 hardcoded events triggered by frame index
- Bottom strip (below video) — 4 metric cells: on-belt / /min / /hr / shift %

**Palette:** Near-black `(14,16,20)` background, amber gold `(0,195,255 BGR)` as hero accent, mint green for positive metrics, coral for alerts. Completely different from construction dashboard.

---

## Shared architecture pattern (both products)

```
Video File → cv2.VideoCapture → InferenceWorker (background thread)
                                        ↓
                                  Grounding DINO (IDEA-Research/grounding-dino-tiny)
                                        ↓
                                  Detections → State object (counts, events, metrics)
                                        ↓
                                  Render functions → 1600×900 OpenCV canvas
                                        ↓
                                  cv2.imshow + optional MP4 export
```

- **Model:** `IDEA-Research/grounding-dino-tiny` (zero-shot, natural language class prompts)
- **Device:** auto-selected MPS → CUDA → CPU
- **Default threshold:** `0.30` for construction, `0.22` for recycling (lower = more detections on dense/messy footage)
- **`infer_every=3`** — inference runs every 3rd frame; `last_detections` is reused between inference calls

## Interactive controls (both products)
- `q` / `ESC` — quit
- `SPACE` — pause/resume
- `r` — reset counters
- `s` — save current frame as PNG
