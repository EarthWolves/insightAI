"""
Grounding DINO — live camera or video file detection with cumulative counters.

Camera usage:
    python camera.py --classes "person, laptop, coffee cup"
    python camera.py --classes "car, bicycle" --threshold 0.35 --camera 1

Video file usage (timelapse-friendly):
    python camera.py --video timelapse.mp4 --classes "person, car"
    python camera.py --video clip.mp4 --classes "cat" --speed 0.25 --save-video

Controls:
    q / ESC  — quit
    r        — reset all counters
    s        — save current frame as PNG
    SPACE    — pause / resume (video mode only)
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
PAD = 12


def draw_counter_overlay(
    frame: np.ndarray, counts: dict[str, int], fps: float = 0.0, progress: float | None = None
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    lines = [("class", "total seen")]
    lines += [(label, str(count)) for label, count in sorted(counts.items())]

    row_h = 28
    col_w = max(cv2.getTextSize(lbl, FONT, 0.6, 1)[0][0] for lbl, _ in lines) + PAD * 4
    panel_w = col_w + 80
    panel_h = len(lines) * row_h + PAD * 2

    # semi-transparent panel (top-left)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    for i, (label, val) in enumerate(lines):
        y = PAD + (i + 1) * row_h - 6
        is_header = i == 0
        color = (180, 180, 180) if is_header else (255, 255, 255)
        thickness = 1 if is_header else 2
        scale = 0.5 if is_header else 0.62
        cv2.putText(out, label, (PAD, y), FONT, scale, color, thickness, cv2.LINE_AA)
        # right-align the count
        (vw, _), _ = cv2.getTextSize(val, FONT, scale, thickness)
        cv2.putText(out, val, (panel_w - vw - PAD, y), FONT, scale, color, thickness, cv2.LINE_AA)

    # FPS or progress — bottom-left
    if progress is not None:
        bar_w = w - PAD * 2
        cv2.rectangle(out, (PAD, h - PAD - 6), (PAD + bar_w, h - PAD + 2), (60, 60, 60), -1)
        cv2.rectangle(out, (PAD, h - PAD - 6), (PAD + int(bar_w * progress), h - PAD + 2), (0, 220, 0), -1)
        cv2.putText(out, f"{progress*100:.0f}%", (PAD, h - PAD - 10), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, f"FPS {fps:.1f}", (PAD, h - PAD), FONT, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
    return out


class InferenceWorker(threading.Thread):
    """
    Runs Grounding DINO inference in a background thread.
    The main thread feeds frames in; this thread writes detections out.
    """

    def __init__(self, processor, model, device, text_prompt: str, threshold: float):
        super().__init__(daemon=True)
        self.processor = processor
        self.model = model
        self.device = device
        self.text_prompt = text_prompt
        self.threshold = threshold

        self._lock = threading.Lock()
        self._input_frame: np.ndarray | None = None
        self._new_detections: list[dict] | None = None  # None = no new result yet
        self._stop_event = threading.Event()

    def submit(self, frame: np.ndarray):
        with self._lock:
            self._input_frame = frame.copy()

    def pop_detections(self) -> list[dict] | None:
        """Return latest detections and clear them, or None if no new result."""
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
                self._input_frame = None  # consume

            if frame is None:
                time.sleep(0.005)
                continue

            # BGR → RGB PIL image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = self.processor(
                images=image, text=self.text_prompt, return_tensors="pt"
            )
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
                detections.append(
                    {
                        "label": label,
                        "score": float(score),
                        "box": [round(v, 2) for v in box.tolist()],
                    }
                )

            with self._lock:
                self._new_detections = detections


def run_video(
    video_path: str,
    classes: str,
    threshold: float = 0.3,
    speed: float = 0.25,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    infer_every: int = 1,
    save_video: bool = False,
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

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # ms to wait between frames at the desired playback speed
    frame_delay_ms = max(1, int(1000 / (src_fps * speed)))

    print(f"Video: {frame_w}x{frame_h} @ {src_fps:.1f} fps  ({total_frames} frames)")
    print(f"Effective display rate: {src_fps * speed:.1f} fps  (delay {frame_delay_ms} ms/frame)\n")

    writer = None
    if save_video:
        out_path = video_path.rsplit(".", 1)[0] + "_counted.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, src_fps * speed, (frame_w, frame_h))
        print(f"Saving annotated video to: {out_path}")

    worker = InferenceWorker(processor, model, device, text_prompt, threshold)
    worker.start()

    frame_idx = 0
    counts: dict[str, int] = defaultdict(int)
    paused = False

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
                    counts[det["label"]] += 1

            progress = frame_idx / max(total_frames, 1)
            display = draw_counter_overlay(frame, counts, progress=progress)

            if writer:
                writer.write(display)

            cv2.imshow("Grounding DINO — video", display)
            frame_idx += 1

        key = cv2.waitKey(frame_delay_ms if not paused else 30) & 0xFF
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


def run_camera(
    classes: str,
    threshold: float = 0.3,
    camera_id: int = 0,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    infer_every: int = 3,
):
    processor, model, device = load_model(model_id)
    text_prompt = format_classes(classes)
    print(f"\nDetecting: {text_prompt}")
    print("Press  q / ESC  to quit,  s  to save a frame.\n")

    worker = InferenceWorker(processor, model, device, text_prompt, threshold)
    worker.start()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: cannot open camera {camera_id}")
        worker.stop()
        return

    frame_idx = 0
    fps = 0.0
    t_prev = time.perf_counter()
    counts: dict[str, int] = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — exiting.")
            break

        # submit every N frames to keep inference from bottlenecking display
        if frame_idx % infer_every == 0:
            worker.submit(frame)

        # accumulate counts whenever a new inference result arrives
        new_dets = worker.pop_detections()
        if new_dets is not None:
            for det in new_dets:
                counts[det["label"]] += 1

        display = draw_counter_overlay(frame, counts, fps)
        cv2.imshow("Grounding DINO — counter", display)

        # FPS calculation
        t_now = time.perf_counter()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("r"):
            counts.clear()
            print("Counters reset.")
        if key == ord("s"):
            path = f"frame_{int(time.time())}.png"
            cv2.imwrite(path, display)
            print(f"Saved: {path}")

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Grounding DINO live camera detection"
    )
    parser.add_argument(
        "--classes", required=True,
        help='Comma-separated classes to detect: "person, cat, dog"',
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--video", default=None,
        help="Path to a video file (MP4 etc.). If omitted, uses live camera.",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0, ignored if --video is set)",
    )
    parser.add_argument(
        "--speed", type=float, default=0.25,
        help="Playback speed multiplier for video mode (default: 0.25 = 4x slower, good for timelapse)",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save annotated output video alongside the source (video mode only)",
    )
    parser.add_argument(
        "--model", default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--infer-every", type=int, default=None,
        help="Run inference every N frames (default: 1 for video, 3 for camera)",
    )
    args = parser.parse_args()

    if args.video:
        run_video(
            video_path=args.video,
            classes=args.classes,
            threshold=args.threshold,
            speed=args.speed,
            model_id=args.model,
            infer_every=args.infer_every if args.infer_every is not None else 1,
            save_video=args.save_video,
        )
    else:
        run_camera(
            classes=args.classes,
            threshold=args.threshold,
            camera_id=args.camera,
            model_id=args.model,
            infer_every=args.infer_every if args.infer_every is not None else 3,
        )


if __name__ == "__main__":
    main()
