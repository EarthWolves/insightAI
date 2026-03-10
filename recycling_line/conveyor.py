"""
RecycleVision — Conveyor belt video detection with cumulative category counters.

Usage:
    python conveyor.py --video "conveyor.mp4"
    python conveyor.py --video "conveyor.mp4" --speed 0.20 --save-video
    python conveyor.py --video "conveyor.mp4" --classes "plastic bottle, aluminum can" --threshold 0.28

Controls:
    q / ESC  — quit
    SPACE    — pause / resume
    r        — reset counters
    s        — save current frame as PNG
"""

import argparse
import threading
import time
from collections import defaultdict
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from detect import format_classes, get_device, load_model

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD  = 12

DEFAULT_CLASSES = "plastic bottle, plastic bag, aluminum can, cardboard box, paper, glass bottle, food waste"

# Map DINO labels → display categories
LABEL_MAP = {
    "plastic bottle":    "plastic",
    "plastic bag":       "plastic",
    "plastic container": "plastic",
    "plastic cup":       "plastic",
    "aluminum can":      "metal",
    "metal can":         "metal",
    "tin can":           "metal",
    "steel can":         "metal",
    "cardboard box":     "paper",
    "cardboard":         "paper",
    "paper":             "paper",
    "newspaper":         "paper",
    "glass bottle":      "glass",
    "glass jar":         "glass",
    "glass container":   "glass",
    "food waste":        "contamination",
    "organic waste":     "contamination",
    "food container":    "contamination",
}

CATEGORY_COLORS = {           # BGR
    "plastic":       (185, 208,  60),
    "paper":         ( 28, 162, 255),
    "metal":         ( 65, 198,  90),
    "glass":         (190,  95, 172),
    "contamination": ( 52,  62, 228),
}


class InferenceWorker(threading.Thread):
    """Runs Grounding DINO inference in a background thread."""

    def __init__(self, processor, model, device, text_prompt: str, threshold: float):
        super().__init__(daemon=True)
        self.processor   = processor
        self.model       = model
        self.device      = device
        self.text_prompt = text_prompt
        self.threshold   = threshold

        self._lock             = threading.Lock()
        self._input_frame      = None
        self._new_detections   = None
        self._stop_event       = threading.Event()

    def submit(self, frame: np.ndarray):
        with self._lock:
            self._input_frame = frame.copy()

    def pop_detections(self):
        with self._lock:
            result = self._new_detections
            self._new_detections = None
            return result

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._input_frame
                self._input_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.threshold,
                target_sizes=[image.size[::-1]],
            )[0]

            detections = []
            for score, label, box in zip(
                results["scores"], results["text_labels"], results["boxes"]
            ):
                detections.append({
                    "label": label,
                    "score": float(score),
                    "box":   [round(v, 2) for v in box.tolist()],
                })

            with self._lock:
                self._new_detections = detections


def draw_overlay(frame: np.ndarray, counts: dict, fps: float = 0.0, progress: float = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Category count panel
    categories = ["plastic", "paper", "metal", "glass", "contamination"]
    lines = [("category", "total")]
    for cat in categories:
        lines.append((cat.upper(), str(counts.get(cat, 0))))

    row_h  = 28
    col_w  = max(cv2.getTextSize(l, FONT, 0.6, 1)[0][0] for l, _ in lines) + PAD * 4
    panel_w = col_w + 80
    panel_h = len(lines) * row_h + PAD * 2

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    for i, (label, val) in enumerate(lines):
        y      = PAD + (i + 1) * row_h - 6
        is_hdr = i == 0
        cat_key = label.lower()
        color  = CATEGORY_COLORS.get(cat_key, (180, 180, 180)) if not is_hdr else (180, 180, 180)
        scale  = 0.5 if is_hdr else 0.62
        thick  = 1 if is_hdr else 2
        cv2.putText(out, label, (PAD, y), FONT, scale, color, thick, cv2.LINE_AA)
        (vw, _), _ = cv2.getTextSize(val, FONT, scale, thick)
        cv2.putText(out, val, (panel_w - vw - PAD, y), FONT, scale, color, thick, cv2.LINE_AA)

    if progress is not None:
        bar_w = w - PAD * 2
        cv2.rectangle(out, (PAD, h - PAD - 6), (PAD + bar_w, h - PAD + 2), (60, 60, 60), -1)
        cv2.rectangle(out, (PAD, h - PAD - 6), (PAD + int(bar_w * progress), h - PAD + 2), (0, 220, 0), -1)
        cv2.putText(out, f"{progress*100:.0f}%", (PAD, h - PAD - 10), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, f"FPS {fps:.1f}", (PAD, h - PAD), FONT, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
    return out


def run_video(
    video_path:  str,
    classes:     str   = DEFAULT_CLASSES,
    threshold:   float = 0.28,
    speed:       float = 0.20,
    model_id:    str   = "IDEA-Research/grounding-dino-tiny",
    infer_every: int   = 1,
    save_video:  bool  = False,
):
    processor, model, device = load_model(model_id)
    text_prompt = format_classes(classes)
    print(f"\nDetecting: {text_prompt}")
    print(f"Playback speed: {speed}x")
    print("Press  q/ESC  to quit,  r  to reset counters,  SPACE  to pause.\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay  = max(1, int(1000 / (src_fps * speed)))

    print(f"Video: {frame_w}x{frame_h} @ {src_fps:.1f}fps  ({total_frames} frames)")
    print(f"Display rate: {src_fps * speed:.1f}fps  ({frame_delay}ms/frame)\n")

    writer = None
    if save_video:
        out_path = video_path.rsplit(".", 1)[0] + "_counted.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                   src_fps * speed, (frame_w, frame_h))
        print(f"Saving to: {out_path}")

    worker    = InferenceWorker(processor, model, device, text_prompt, threshold)
    worker.start()

    frame_idx = 0
    counts: dict = defaultdict(int)
    paused    = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"\nEnd of video. Final counts: {dict(counts)}")
                break

            if frame_idx % infer_every == 0:
                worker.submit(frame)

            new_dets = worker.pop_detections()
            if new_dets is not None:
                for det in new_dets:
                    cat = LABEL_MAP.get(det["label"], "plastic")
                    counts[cat] += 1

            progress = frame_idx / max(total_frames, 1)
            display  = draw_overlay(frame, counts, progress=progress)

            if writer:
                writer.write(display)

            cv2.imshow("RecycleVision — Conveyor Detection", display)
            frame_idx += 1

        key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            counts.clear()
            print("Counters reset.")
        if key == ord("s"):
            path = f"frame_{frame_idx}.png"
            cv2.imwrite(path, display)
            print(f"Saved: {path}")
        if key == ord(" "):
            paused = not paused
            print("Paused." if paused else "Resumed.")

    worker.stop()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="RecycleVision conveyor belt detection")
    parser.add_argument("--video",       required=True)
    parser.add_argument("--classes",     default=DEFAULT_CLASSES)
    parser.add_argument("--threshold",   type=float, default=0.28)
    parser.add_argument("--speed",       type=float, default=0.20,
                        help="Playback speed (default 0.20 = 5x slower)")
    parser.add_argument("--model",       default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--infer-every", type=int, default=1)
    parser.add_argument("--save-video",  action="store_true")
    args = parser.parse_args()

    run_video(
        video_path  = args.video,
        classes     = args.classes,
        threshold   = args.threshold,
        speed       = args.speed,
        model_id    = args.model,
        infer_every = args.infer_every,
        save_video  = args.save_video,
    )


if __name__ == "__main__":
    main()
