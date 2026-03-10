"""
RecycleVision Counter — Conveyor belt object counting with pluggable detectors.

Architecture: Strategy pattern with BaseDetector ABC.
  - YOLODetector  : any ultralytics-compatible .pt model (auto-downloads fallback)
  - MOG2Detector  : background subtraction + HSV heuristics (no model required)

Counting: CentroidTracker (nearest-centroid) + virtual line crossing.
"""

import os
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import OrderedDict, deque

import cv2
import numpy as np


# ── Shared category state — populated by whichever detector is initialised ────
# key → (DISPLAY_LABEL, BGR_colour)
CATEGORIES:    dict = {}
CATEGORY_KEYS: list = []
DEFAULT_CAT         = "unknown"

# Colour palette assigned by class index (for YOLO dynamic classes)
_PALETTE = [
    (  0, 195, 255),   # amber gold
    ( 48, 155,  88),   # olive green
    ( 42, 100, 165),   # warm brown
    (200, 200,  60),   # aqua/cyan
    (195, 185, 145),   # steel grey
    (110, 225, 140),   # mint green
    ( 80,  88, 225),   # coral
    (220, 155,  55),   # sky blue
    (175, 225,  75),   # teal
    (220, 210, 190),   # off-white
]


def _set_categories(cats: dict):
    """Replace global CATEGORIES / CATEGORY_KEYS in-place."""
    global DEFAULT_CAT
    CATEGORIES.clear()
    CATEGORIES.update(cats)
    CATEGORY_KEYS.clear()
    CATEGORY_KEYS.extend(CATEGORIES.keys())
    DEFAULT_CAT = CATEGORY_KEYS[0] if CATEGORY_KEYS else "unknown"


# ── Base Detector (Strategy interface) ────────────────────────────────────────

class BaseDetector(ABC):
    """
    Common interface for all detection backends.

    `apply(frame)` must return:
      bboxes — [(x1, y1, x2, y2), …]   full-frame pixel coords
      labels — [str, …]                  parallel lowercase category keys
    """

    @abstractmethod
    def apply(self, frame: np.ndarray) -> tuple[list, list]:
        ...

    @property
    @abstractmethod
    def detector_name(self) -> str:
        ...


# ── YOLOv8 Detector ───────────────────────────────────────────────────────────

_FALLBACK_URL  = "https://github.com/gianlucasposito/YOLO-Waste-Detection/raw/main/best_model.pt"
_FALLBACK_PATH = "best_model.pt"


def _resolve_weights(model_path: str) -> str:
    """Return model_path if it exists; otherwise download the fallback model."""
    if os.path.isfile(model_path):
        return model_path
    print(f"[YOLODetector] '{model_path}' not found.")
    if not os.path.isfile(_FALLBACK_PATH):
        print(f"[YOLODetector] Downloading fallback waste-classification model ...")
        print(f"               {_FALLBACK_URL}")
        urllib.request.urlretrieve(_FALLBACK_URL, _FALLBACK_PATH)
        print(f"[YOLODetector] Saved to '{_FALLBACK_PATH}'")
    else:
        print(f"[YOLODetector] Using cached fallback '{_FALLBACK_PATH}'")
    return _FALLBACK_PATH


class YOLODetector(BaseDetector):
    """
    YOLOv8 detection + classification via ultralytics.

    Any ultralytics-compatible .pt weights work (different models = different
    class sets; CATEGORIES is rebuilt from model.names at load time).

    If the specified weights file is missing, auto-downloads:
      gianlucasposito/YOLO-Waste-Detection  (Glass, Metal, Paper, Plastic, Waste)

    Args:
        model_path:  path to .pt weights file  (default: "best.pt")
        conf:        confidence threshold       (default: 0.25)
        infer_every: run inference every N frames, reuse result in-between
    """

    def __init__(self, model_path: str = "best.pt", conf: float = 0.25,
                 infer_every: int = 2):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        import torch
        if torch.backends.mps.is_available():   device = "mps"
        elif torch.cuda.is_available():          device = "cuda"
        else:                                    device = "cpu"

        resolved     = _resolve_weights(model_path)
        print(f"[YOLODetector] loading '{resolved}' on {device}")
        self.model   = YOLO(resolved)
        self.model.to(device)

        self.conf        = conf
        self.infer_every = infer_every
        self._frame_ctr  = 0
        self._last_bboxes: list = []
        self._last_labels: list = []

        # Build category map from model class names
        self._names = {k: v.lower() for k, v in self.model.names.items()}
        cats = {
            v.lower(): (v.upper()[:9], _PALETTE[k % len(_PALETTE)])
            for k, v in self.model.names.items()
        }
        _set_categories(cats)
        print(f"[YOLODetector] classes: {list(self._names.values())}")

    @property
    def detector_name(self) -> str:
        return "yolo"

    def apply(self, frame: np.ndarray) -> tuple[list, list]:
        if self._frame_ctr % self.infer_every == 0:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            bboxes, labels = [], []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_name = self._names.get(int(box.cls[0]), DEFAULT_CAT)
                if cls_name not in CATEGORIES:
                    cls_name = DEFAULT_CAT
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                labels.append(cls_name)
            self._last_bboxes = bboxes
            self._last_labels = labels
        self._frame_ctr += 1
        return self._last_bboxes, self._last_labels


# ── MOG2 Detector (background subtraction + HSV heuristics) ──────────────────

_MOG2_CATEGORIES = {
    "crv":     ("CRV",     (175, 225,  75)),   # seafoam — plastic/CRV
    "compost": ("COMPOST", ( 48, 155,  88)),   # olive   — organic
    "metal":   ("METAL",   (195, 185, 145)),   # steel   — metal
}


def _classify_blob_hsv(frame: np.ndarray, bbox: tuple) -> str:
    """HSV colour heuristics: metal → compost → crv (default)."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    fh, fw = frame.shape[:2]
    roi = frame[max(0, y1): min(fh, y2), max(0, x1): min(fw, x2)]
    if roi.size < 50:
        return "crv"
    px  = roi.reshape(1, -1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v   = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    mean_s    = float(np.mean(s))
    mean_v    = float(np.mean(v))
    mean_h    = float(np.mean(h))
    pct_dark  = float(np.mean(v < 80))
    if mean_s < 52 and mean_v > 72 and pct_dark < 0.35:
        return "metal"
    if pct_dark > 0.50:
        return "compost"
    if 5 <= mean_h <= 24 and mean_s > 38:
        return "compost"
    return "crv"


class MOG2Detector(BaseDetector):
    """
    Background subtraction (MOG2) + HSV colour heuristics.

    No ML model required — works on any hardware in real-time.
    Returns bboxes AND filled contours for the segmentation overlay.

    Categories: CRV (plastic) · COMPOST (organic) · METAL

    Args:
        history:       MOG2 history length  (default 500)
        var_threshold: MOG2 sensitivity — lower = more detections (default 40)
        min_area:      min blob area in px²  (default 800)
        max_area:      max blob area in px²  (default 60000)
        infer_every:   apply MOG2 every N frames
    """

    def __init__(self, history: int = 500, var_threshold: float = 40.0,
                 min_area: int = 800, max_area: int = 60_000,
                 infer_every: int = 1):
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.min_area    = min_area
        self.max_area    = max_area
        self.infer_every = infer_every
        self._frame_ctr  = 0
        self._k5         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k11        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._last_bboxes:    list = []
        self._last_labels:    list = []
        self._last_contours:  list = []   # raw contours (for overlay drawing)
        _set_categories(_MOG2_CATEGORIES)
        print("[MOG2Detector] ready — categories: CRV · COMPOST · METAL")

    @property
    def detector_name(self) -> str:
        return "mog2"

    @property
    def last_contours(self) -> list:
        """Raw contours parallel to last bboxes (used by dashboard for fills)."""
        return self._last_contours

    def apply(self, frame: np.ndarray) -> tuple[list, list]:
        # MOG2 must see every frame to build a good background model,
        # but we still honour infer_every for the bbox/label outputs.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        mask = self.sub.apply(blur)

        if self._frame_ctr % self.infer_every == 0:
            _, mask_th = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask_th = cv2.morphologyEx(mask_th, cv2.MORPH_OPEN,  self._k5)
            mask_th = cv2.morphologyEx(mask_th, cv2.MORPH_CLOSE, self._k11, iterations=2)
            mask_th = cv2.dilate(mask_th, self._k5, iterations=2)

            all_cnt, _ = cv2.findContours(mask_th, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            bboxes, labels, contours = [], [], []
            for cnt in all_cnt:
                area = cv2.contourArea(cnt)
                if self.min_area <= area <= self.max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox = (x, y, x + w, y + h)
                    bboxes.append(bbox)
                    labels.append(_classify_blob_hsv(frame, bbox))
                    contours.append(cnt)
            self._last_bboxes   = bboxes
            self._last_labels   = labels
            self._last_contours = contours

        self._frame_ctr += 1
        return self._last_bboxes, self._last_labels


# ── Detector Factory ──────────────────────────────────────────────────────────

def build_detector(detector_type: str, **kwargs) -> BaseDetector:
    """
    Factory function for detector backends.

    detector_type:
      "yolo"  → YOLODetector(**kwargs)   kwargs: model_path, conf, infer_every
      "mog2"  → MOG2Detector(**kwargs)   kwargs: var_threshold, min_area, infer_every

    Example:
        det = build_detector("yolo", model_path="best.pt", conf=0.3)
        det = build_detector("mog2", var_threshold=30, min_area=600)
    """
    registry = {
        "yolo": YOLODetector,
        "mog2": MOG2Detector,
    }
    if detector_type not in registry:
        raise ValueError(f"Unknown detector '{detector_type}'. Choose from: {list(registry)}")
    return registry[detector_type](**kwargs)


# ── Centroid Tracker ──────────────────────────────────────────────────────────

class CentroidTracker:
    """Nearest-centroid assignment. Stores bbox and class label per tracked ID."""

    def __init__(self, max_disappeared: int = 25, max_distance: int = 120):
        self.next_id         = 0
        self.objects         = OrderedDict()   # id → (cx, cy)
        self.bboxes          = OrderedDict()   # id → (x1, y1, x2, y2)
        self.labels          = OrderedDict()   # id → str class label
        self.disappeared     = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def register(self, centroid, bbox, label=None):
        label = label or DEFAULT_CAT
        self.objects[self.next_id]     = centroid
        self.bboxes[self.next_id]      = bbox
        self.labels[self.next_id]      = label
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        self.bboxes.pop(obj_id, None)
        self.labels.pop(obj_id, None)

    def update(self, bboxes: list, labels: list = None) -> dict:
        if labels is None:
            labels = [DEFAULT_CAT] * len(bboxes)

        if not bboxes:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        centroids = [(int((x1 + x2) / 2), int((y1 + y2) / 2))
                     for x1, y1, x2, y2 in bboxes]

        if not self.objects:
            for c, b, l in zip(centroids, bboxes, labels):
                self.register(c, b, l)
            return self.objects

        ids   = list(self.objects.keys())
        old_c = list(self.objects.values())

        D = np.linalg.norm(
            np.array(old_c)[:, None] - np.array(centroids)[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_r, used_c = set(), set()
        for r, c in zip(rows, cols):
            if r in used_r or c in used_c:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = ids[r]
            self.objects[oid]     = centroids[c]
            self.bboxes[oid]      = bboxes[c]
            self.labels[oid]      = labels[c]
            self.disappeared[oid] = 0
            used_r.add(r)
            used_c.add(c)

        for r in set(range(len(ids))) - used_r:
            oid = ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for c in set(range(len(centroids))) - used_c:
            self.register(centroids[c], bboxes[c], labels[c])

        return self.objects


# ── Count State ───────────────────────────────────────────────────────────────

class CountState:
    """Tracks total count, per-category breakdown, throughput, and belt status."""

    EVENTS_DATA = [
        (50,   "INFO", "08:05", "Belt active — detector warming up"),
        (200,  "INFO", "08:21", "Detection active — counting started"),
        (450,  "WARN", "09:09", "Dense cluster — possible jam risk"),
        (750,  "INFO", "09:34", "Flow rate above shift target"),
        (1100, "WARN", "10:01", "High item density — monitor throughput"),
        (1500, "CRIT", "10:28", "Throughput dip — check belt speed"),
        (2000, "INFO", "11:41", "Pace on target — shift 47% complete"),
        (2600, "INFO", "12:35", "Belt speed stable — optimal flow"),
    ]

    def __init__(self, warmup_frames: int = 10):
        self.total           = 0
        self.category_counts = {k: 0 for k in CATEGORY_KEYS}
        self._prev_x         = {}
        self._cross_times    = deque()
        self.on_belt         = 0
        self.event_log       = deque(maxlen=5)
        self._triggered      = set()
        self.pulse_frame     = -999
        self.warmup_frames   = warmup_frames

    def update(self, objects: dict, categories: dict, line_x: int,
               frame_idx: int) -> int:
        """Returns number of items that crossed the counting line this frame."""
        crossed = 0
        if frame_idx >= self.warmup_frames:
            for oid, (cx, cy) in objects.items():
                prev_x = self._prev_x.get(oid)
                if prev_x is not None and prev_x < line_x <= cx:
                    crossed += 1
                    cat = categories.get(oid, DEFAULT_CAT)
                    if cat in self.category_counts:
                        self.category_counts[cat] += 1

        for oid in list(self._prev_x):
            if oid not in objects:
                del self._prev_x[oid]
        for oid, (cx, cy) in objects.items():
            self._prev_x[oid] = cx

        self.total   += crossed
        self.on_belt  = len(objects)
        if crossed:
            self.pulse_frame = frame_idx
            now = time.perf_counter()
            self._cross_times.extend([now] * crossed)

        now = time.perf_counter()
        while self._cross_times and now - self._cross_times[0] > 30:
            self._cross_times.popleft()

        for i, (trigger, sev, ts, msg) in enumerate(self.EVENTS_DATA):
            if i not in self._triggered and frame_idx >= trigger:
                self._triggered.add(i)
                self.event_log.appendleft((ts, sev, msg))

        return crossed

    @property
    def items_per_min(self) -> float:
        return len(self._cross_times) * 2.0

    @property
    def items_per_hour(self) -> int:
        return int(self.items_per_min * 60)

    @property
    def belt_load(self) -> tuple:
        n = self.on_belt
        if n == 0:  return "IDLE",   (55,  65,  80)
        if n <= 3:  return "SPARSE", (220, 155,  55)
        if n <= 8:  return "NORMAL", (110, 225, 140)
        if n <= 15: return "DENSE",  (0,  195, 255)
        return      "HIGH",          (80,  88, 225)
