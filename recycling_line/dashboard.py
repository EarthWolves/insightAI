"""
RecycleVision — Conveyor Belt Counter Dashboard

Pluggable detection backend:
  --detector yolo   YOLOv8 model  (default; auto-downloads fallback weights)
  --detector mog2   Background subtraction + HSV heuristics (no model needed)

Usage:
    python dashboard.py --video conveyor.mp4
    python dashboard.py --video conveyor.mp4 --detector yolo --model best.pt
    python dashboard.py --video conveyor.mp4 --detector mog2 --threshold 30
    python dashboard.py --video conveyor.mp4 --detector yolo --conf 0.30 --roi 80,60,920,500

Controls:
    q / ESC     — quit
    SPACE       — pause / resume
    r           — reset count
    +  /  -     — shift counting line right / left (10px per press)
    s           — save current frame
"""

import argparse
import time

import cv2
import numpy as np

import counter as _c
from counter import build_detector, CentroidTracker, CountState, CATEGORIES, CATEGORY_KEYS

# ── Canvas ─────────────────────────────────────────────────────────────────────
CW, CH   = 1600, 900
HDR_H    = 56
G        = 8

BODY_Y   = HDR_H + G
BODY_H   = CH - BODY_Y - G

LW       = 400
VX       = G + LW + G
VW       = CW - VX - G
VH       = VW * 9 // 16

STRIP_Y  = BODY_Y + VH + G
STRIP_H  = CH - STRIP_Y - G

SHIFT_TARGET = 5_000

S_COUNT  = 310
S_THRU   = 130
S_SHIFT  = 110
S_BELT   = 108
S_EVENTS = BODY_H - S_COUNT - S_THRU - S_SHIFT - S_BELT - 4 * G

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
BG    = ( 14,  16,  20)
PNL   = ( 19,  22,  28)
HDR_C = (  9,  11,  15)
SEP   = ( 30,  38,  52)
DIM   = ( 50,  58,  76)
WHT   = (248, 252, 255)
GRY   = (148, 162, 185)
MUTED = ( 82,  94, 118)
GOLD  = (  0, 195, 255)
MINT  = (110, 225, 140)
CORAL = ( 80,  88, 225)
BLUE  = (220, 155,  55)

F0 = cv2.FONT_HERSHEY_SIMPLEX
F1 = cv2.FONT_HERSHEY_DUPLEX

SEG_ALPHA = 0.45


# ── Helpers ────────────────────────────────────────────────────────────────────

def txt(c, s, x, y, color=WHT, scale=0.50, thick=1, font=F0):
    cv2.putText(c, str(s), (x, y), font, scale, color, thick, cv2.LINE_AA)

def hbar(c, x, y, w, h, pct, color, bg=DIM):
    cv2.rectangle(c, (x, y), (x + w, y + h), bg, -1)
    fw = max(0, int(w * min(pct, 100.0) / 100))
    if fw:
        cv2.rectangle(c, (x, y), (x + fw, y + h), color, -1)

def hsep(c, y, x1=None, x2=None):
    cv2.line(c, (x1 or G, y), (x2 or G + LW, y), SEP, 1)

def sev_clr(sev):
    return {"INFO": BLUE, "WARN": GOLD, "CRIT": CORAL}.get(sev, GRY)

def sim_time(frame_idx, total):
    mins = int(frame_idx / max(total, 1) * 480)
    return f"{8 + mins // 60:02d}:{mins % 60:02d}"

def pulse_alpha(frame_idx, pulse_frame, duration=22):
    elapsed = frame_idx - pulse_frame
    if elapsed < 0 or elapsed >= duration:
        return 0.0
    return 1.0 - elapsed / duration


# ── Header ─────────────────────────────────────────────────────────────────────

def render_header(c, detector_name, frame_idx, total_frames):
    cv2.rectangle(c, (0, 0), (CW, HDR_H), HDR_C, -1)
    cv2.line(c, (0, HDR_H - 1), (CW, HDR_H - 1), SEP, 1)
    cv2.rectangle(c, (0, 0), (CW, 3), GOLD, -1)

    txt(c, "RECYCLEVISION", 20, 36, GOLD, 0.70, 2, F1)
    txt(c, "GreenSort Facility — Line 3  //  Day Shift 08:00–16:00", 202, 20, WHT, 0.42, 1)

    backend = "YOLOv8 Waste Classification" if detector_name == "yolo" \
              else "Background Subtraction + HSV Heuristics"
    txt(c, f"Conveyor Belt Object Counter  •  {backend}  •  Real-Time",
        202, 38, GRY, 0.36, 1)

    pct = frame_idx / max(total_frames, 1)
    txt(c, f"SIM {sim_time(frame_idx, total_frames)}   VIDEO {pct*100:.0f}%",
        CW - 200, 32, MUTED, 0.38, 1)


# ── Left Stats Panel ───────────────────────────────────────────────────────────

def render_left_panel(c, state: CountState, frame_idx: int):
    x, y, w = G, BODY_Y, LW
    px = x + 22
    pw = w - 44

    cv2.rectangle(c, (x, y), (x + w, CH - G), PNL, -1)
    cv2.line(c, (x + w, y), (x + w, CH - G), SEP, 1)

    # ── Hero count ────────────────────────────────────────────────────────────
    sy, sh = y, S_COUNT

    txt(c, "ITEMS COUNTED", px, sy + 28, GRY, 0.36, 1)

    count_str = f"{state.total:,}"
    (nw, nh), _ = cv2.getTextSize(count_str, F1, 3.2, 4)
    nx = x + (w - nw) // 2
    ny = sy + 48 + nh

    pa = pulse_alpha(frame_idx, state.pulse_frame, duration=25)
    if pa > 0.05:
        ov = c.copy()
        pad = 12
        cv2.rectangle(ov, (nx - pad, ny - nh - pad), (nx + nw + pad, ny + pad), GOLD, -1)
        cv2.addWeighted(ov, 0.07 * pa, c, 1 - 0.07 * pa, 0, c)

    txt(c, count_str, nx, ny, GOLD, 3.2, 4, F1)

    sub = "items recorded this session"
    (sw, _), _ = cv2.getTextSize(sub, F0, 0.36, 1)
    txt(c, sub, x + (w - sw) // 2, ny + 18, MUTED, 0.36, 1)

    # ── Category breakdown (dynamic grid) ────────────────────────────────────
    cat_y = ny + 34
    cv2.line(c, (px, cat_y), (px + pw, cat_y), SEP, 1)
    cat_y += 10

    n_cats  = max(len(CATEGORY_KEYS), 1)
    n_cols  = min(3, n_cats)
    n_rows  = (n_cats + n_cols - 1) // n_cols
    col_w   = pw // n_cols
    total_c = max(state.total, 1)
    row_h   = 60

    for i, key in enumerate(CATEGORY_KEYS):
        label, clr = CATEGORIES[key]
        count      = state.category_counts.get(key, 0)
        pct        = count / total_c * 100
        row        = i // n_cols
        col        = i %  n_cols
        cx_        = px + col * col_w
        cy_        = cat_y + row * row_h

        cv2.circle(c, (cx_ + 6, cy_ + 7), 4, clr, -1)
        txt(c, label[:8], cx_ + 14, cy_ + 11, clr, 0.34, 1)

        (vw, vh), _ = cv2.getTextSize(f"{count:,}", F1, 0.72, 2)
        txt(c, f"{count:,}", cx_, cy_ + 26 + vh, clr, 0.72, 2, F1)

        hbar(c, cx_, cy_ + 30 + vh, col_w - 6, 3, pct, clr)
        txt(c, f"{pct:.0f}%", cx_, cy_ + 43 + vh, MUTED, 0.30, 1)

        if col < n_cols - 1:
            cv2.line(c, (cx_ + col_w, cat_y),
                     (cx_ + col_w, cat_y + row_h * n_rows), SEP, 1)

    hsep(c, sy + sh - 1)

    # ── Throughput ────────────────────────────────────────────────────────────
    sy += sh + G;  sh = S_THRU
    txt(c, "THROUGHPUT", px, sy + 26, GRY, 0.36, 1)
    rate = state.items_per_min
    hr   = state.items_per_hour
    hw   = pw // 2 - 8
    (rw, rh), _ = cv2.getTextSize(f"{rate:.1f}", F1, 1.15, 2)
    txt(c, f"{rate:.1f}", px,          sy + 54 + rh, MINT, 1.15, 2, F1)
    txt(c, "per min",     px,          sy + 62 + rh + 10, MUTED, 0.35, 1)
    hx2 = px + hw + 16
    txt(c, f"{hr:,}",          hx2, sy + 54 + rh, MINT, 1.15, 2, F1)
    txt(c, "projected / hr",   hx2, sy + 62 + rh + 10, MUTED, 0.35, 1)
    hsep(c, sy + sh - 1)

    # ── Shift target ──────────────────────────────────────────────────────────
    sy += sh + G;  sh = S_SHIFT
    txt(c, "SHIFT TARGET", px, sy + 26, GRY, 0.36, 1)
    sp     = min(100.0, state.total / SHIFT_TARGET * 100)
    sp_clr = MINT if sp < 80 else GOLD
    (tw, th), _ = cv2.getTextSize(f"{state.total:,}", F1, 0.88, 2)
    txt(c, f"{state.total:,}", px, sy + 54 + th, sp_clr, 0.88, 2, F1)
    txt(c, f"/ {SHIFT_TARGET:,}", px + tw + 8, sy + 54 + th, GRY, 0.44, 1)
    hbar(c, px, sy + 68 + th, pw, 7, sp, sp_clr)
    txt(c, f"{sp:.1f}%", px + pw + 6, sy + 68 + th + 6, sp_clr, 0.36, 1)
    hsep(c, sy + sh - 1)

    # ── Belt status ───────────────────────────────────────────────────────────
    sy += sh + G;  sh = S_BELT
    txt(c, "BELT STATUS", px, sy + 26, GRY, 0.36, 1)
    bl_label, bl_clr = state.belt_load
    txt(c, f"{state.on_belt}", px,      sy + 70, bl_clr, 1.30, 2, F1)
    txt(c, "on belt now",      px + 54, sy + 60, GRY,    0.36, 1)
    txt(c, bl_label,           px + 54, sy + 78, bl_clr, 0.48, 2)
    hsep(c, sy + sh - 1)

    # ── Events ────────────────────────────────────────────────────────────────
    sy += sh + G
    txt(c, "EVENTS", px, sy + 22, GRY, 0.36, 1)
    events = list(state.event_log)
    if not events:
        txt(c, "Waiting for detections...", px, sy + 48, MUTED, 0.38, 1)
    else:
        avail_h = CH - G - (sy + 30)
        rh      = max(34, avail_h // max(len(events), 1))
        for i, (ts, sev, msg) in enumerate(events):
            ry = sy + 30 + i * rh
            if ry + rh > CH - G - 4:
                break
            sc = sev_clr(sev)
            cv2.circle(c, (px + 4, ry + 8), 4, sc, -1)
            txt(c, ts,  px + 15, ry + 11, MUTED, 0.36, 1)
            txt(c, sev, px + 60, ry + 11, sc,    0.34, 1)
            mc = (pw - 20) // 7
            txt(c, msg if len(msg) <= mc else msg[:mc - 1] + "...",
                px + 15, ry + 25, GRY, 0.37, 1)


# ── Video Panel ────────────────────────────────────────────────────────────────

def render_video(c, frame, det_items, tracked_centroids, state, line_x_raw,
                 frame_idx, fps, use_contours=False):
    """
    det_items:
      YOLO  — [(bbox_raw, label), …]
      MOG2  — [(contour_raw, label), …]  when use_contours=True
    """
    orig_h, orig_w = frame.shape[:2]
    vf = cv2.resize(frame, (VW, VH))
    sx = VW / orig_w
    sy = VH / orig_h

    # ── Detection overlay ─────────────────────────────────────────────────────
    if det_items:
        overlay = vf.copy()

        if use_contours:
            # MOG2 path: filled contour polygons
            for cnt_raw, cat in det_items:
                _, clr = CATEGORIES.get(cat, ("?", GOLD))
                cnt_d = (cnt_raw.astype(np.float32)
                         * np.array([[[sx, sy]]])).astype(np.int32)
                cv2.drawContours(overlay, [cnt_d], -1, clr, cv2.FILLED)
            cv2.addWeighted(overlay, SEG_ALPHA, vf, 1 - SEG_ALPHA, 0, vf)
            for cnt_raw, cat in det_items:
                _, clr = CATEGORIES.get(cat, ("?", GOLD))
                cnt_d = (cnt_raw.astype(np.float32)
                         * np.array([[[sx, sy]]])).astype(np.int32)
                cv2.drawContours(vf, [cnt_d], -1, clr, 2)
        else:
            # YOLO path: filled bounding boxes
            for (x1, y1, x2, y2), cat in det_items:
                _, clr = CATEGORIES.get(cat, ("?", GOLD))
                cv2.rectangle(overlay,
                              (int(x1*sx), int(y1*sy)),
                              (int(x2*sx), int(y2*sy)), clr, -1)
            cv2.addWeighted(overlay, SEG_ALPHA, vf, 1 - SEG_ALPHA, 0, vf)
            for (x1, y1, x2, y2), cat in det_items:
                label, clr = CATEGORIES.get(cat, ("?", GOLD))
                dx1, dy1 = int(x1*sx), int(y1*sy)
                dx2, dy2 = int(x2*sx), int(y2*sy)
                cv2.rectangle(vf, (dx1, dy1), (dx2, dy2), clr, 2)
                (lw, lh), _ = cv2.getTextSize(label, F0, 0.38, 1)
                cv2.rectangle(vf, (dx1, dy1 - lh - 6), (dx1 + lw + 6, dy1), clr, -1)
                txt(vf, label, dx1 + 3, dy1 - 4, BG, 0.38, 1)

    # ── Centroid dots ─────────────────────────────────────────────────────────
    for cx_r, cy_r, cat in tracked_centroids:
        _, clr = CATEGORIES.get(cat, ("?", GOLD))
        cv2.circle(vf, (int(cx_r*sx), int(cy_r*sy)), 5, clr, -1)
        cv2.circle(vf, (int(cx_r*sx), int(cy_r*sy)), 8, clr, 1)

    # ── Counting line ─────────────────────────────────────────────────────────
    line_x_d = int(line_x_raw * sx)
    pa = pulse_alpha(frame_idx, state.pulse_frame, duration=22)
    if pa > 0.02:
        glow = vf.copy()
        cv2.line(glow, (line_x_d, 0), (line_x_d, VH), WHT, int(4 + 18 * pa))
        cv2.addWeighted(glow, 0.40 * pa, vf, 1 - 0.40 * pa, 0, vf)
    for seg_y in range(0, VH, 40):
        cv2.line(vf, (line_x_d, seg_y), (line_x_d, min(seg_y + 20, VH)), GOLD, 2)
    for arrow_y in range(VH // 6, VH, VH // 4):
        pts = np.array([[line_x_d - 1, arrow_y - 7],
                        [line_x_d + 9, arrow_y],
                        [line_x_d - 1, arrow_y + 7]], dtype=np.int32)
        cv2.fillPoly(vf, [pts], GOLD)
    lbl = "COUNT LINE"
    (lw, _), _ = cv2.getTextSize(lbl, F0, 0.33, 1)
    cv2.rectangle(vf, (line_x_d + 6, 6), (line_x_d + 6 + lw + 8, 22), PNL, -1)
    txt(vf, lbl, line_x_d + 10, 19, GOLD, 0.33, 1)

    # ── FPS badge ─────────────────────────────────────────────────────────────
    cv2.rectangle(vf, (6, 4), (76, 20), PNL, -1)
    txt(vf, f"{fps:.0f} FPS", 10, 17, MINT, 0.36, 1)

    c[BODY_Y: BODY_Y + VH, VX: VX + VW] = vf
    cv2.rectangle(c, (VX, BODY_Y), (VX + VW, BODY_Y + VH), SEP, 1)


# ── Bottom Metric Strip ────────────────────────────────────────────────────────

def render_strip(c, state: CountState):
    y, h = STRIP_Y, STRIP_H
    cv2.rectangle(c, (VX, y), (VX + VW, y + h), PNL, -1)
    cv2.line(c, (VX, y), (VX + VW, y), SEP, 1)
    cells = [
        ("ON BELT NOW",  str(state.on_belt),                                  GRY),
        ("ITEMS / MIN",  f"{state.items_per_min:.1f}",                        MINT),
        ("ITEMS / HOUR", f"{state.items_per_hour:,}",                         MINT),
        ("SHIFT",        f"{min(100, state.total/SHIFT_TARGET*100):.1f}%",    GOLD),
    ]
    cw = VW // len(cells)
    for i, (label, val, clr) in enumerate(cells):
        cx_ = VX + i * cw
        if i > 0:
            cv2.line(c, (cx_, y + 10), (cx_, y + h - 10), SEP, 1)
        (vw, vh), _ = cv2.getTextSize(val,   F1, 0.92, 2)
        (lw, _),  _ = cv2.getTextSize(label, F0, 0.36, 1)
        txt(c, label, cx_ + (cw - lw) // 2, y + 36,      GRY, 0.36, 1)
        txt(c, val,   cx_ + (cw - vw) // 2, y + h - 30,  clr, 0.92, 2, F1)


# ── Main Loop ──────────────────────────────────────────────────────────────────

def parse_roi(s):
    if not s:
        return None
    try:
        vals = tuple(int(v) for v in s.split(","))
        assert len(vals) == 4
        return vals
    except Exception:
        print(f"Warning: invalid --roi '{s}', ignoring.")
        return None


def run(
    video_path:    str,
    detector_type: str   = "yolo",
    model_path:    str   = "best.pt",
    speed:         float = 0.5,
    line_pos:      float = 0.55,
    roi:           tuple = None,
    conf:          float = 0.25,
    infer_every:   int   = 2,
    var_threshold: float = 40.0,
    min_area:      int   = 800,
    save_video:    bool  = False,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open '{video_path}'")
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms     = max(1, int(1000 / (src_fps * speed)))
    line_x_raw   = (roi[0] + int((roi[2] - roi[0]) * line_pos)) if roi \
                   else int(orig_w * line_pos)

    # Build the chosen detector
    if detector_type == "yolo":
        detector = build_detector("yolo", model_path=model_path,
                                  conf=conf, infer_every=infer_every)
    else:
        detector = build_detector("mog2", var_threshold=var_threshold,
                                  min_area=min_area, infer_every=infer_every)

    use_contours = (detector.detector_name == "mog2")

    tracker = CentroidTracker(max_disappeared=25, max_distance=130)
    state   = CountState(warmup_frames=10 if detector_type == "yolo" else 60)

    print(f"Video: {orig_w}x{orig_h} @ {src_fps:.0f}fps  |  {total_frames} frames  |  speed {speed}x")
    print(f"Detector: {detector_type.upper()}  |  Line x={line_x_raw}px")
    print("Controls: q/ESC quit  SPACE pause  r reset  +/- move line  s save\n")

    writer = None
    if save_video:
        out_path = video_path.rsplit(".", 1)[0] + f"_{detector_type}_counted.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                   src_fps * speed, (CW, CH))
        print(f"Saving to: {out_path}")

    frame_idx      = 0
    paused         = False
    fps            = 0.0
    t_prev         = time.perf_counter()
    last_canvas    = None
    last_det_items = []
    last_centroids = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"\nEnd of video.  Total counted: {state.total}")
                for key in CATEGORY_KEYS:
                    print(f"  {CATEGORIES[key][0]}: {state.category_counts.get(key, 0)}")
                break

            infer_frame = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame
            bboxes, labels = detector.apply(infer_frame)

            if roi:
                bboxes = [(bx1+roi[0], by1+roi[1], bx2+roi[0], by2+roi[1])
                          for bx1, by1, bx2, by2 in bboxes]

            objects    = tracker.update(bboxes, labels)
            categories = {oid: tracker.labels[oid]
                          for oid in objects if oid in tracker.labels}
            state.update(dict(objects), categories, line_x_raw, frame_idx)

            t_now  = time.perf_counter()
            fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            if use_contours:
                # MOG2: pair contours (not bboxes) with labels for overlay
                contours = detector.last_contours
                if roi:
                    contours = [cnt + np.array([[[roi[0], roi[1]]]])
                                for cnt in contours]
                last_det_items = list(zip(contours, labels))
            else:
                last_det_items = list(zip(bboxes, labels))

            last_centroids = [(cx, cy, categories.get(oid, _c.DEFAULT_CAT))
                              for oid, (cx, cy) in objects.items()]

            canvas = np.full((CH, CW, 3), BG, dtype=np.uint8)
            render_header(canvas, detector.detector_name, frame_idx, total_frames)
            render_left_panel(canvas, state, frame_idx)
            render_video(canvas, frame, last_det_items, last_centroids, state,
                         line_x_raw, frame_idx, fps, use_contours=use_contours)
            render_strip(canvas, state)

            last_canvas = canvas
            if writer:
                writer.write(canvas)
            frame_idx += 1

        if last_canvas is not None:
            cv2.imshow("RecycleVision — Belt Counter", last_canvas)

        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            print("Paused." if paused else "Resumed.")
        if key == ord("r"):
            state.total = 0
            state.category_counts = {k: 0 for k in CATEGORY_KEYS}
            state._prev_x.clear()
            state._cross_times.clear()
            print("Count reset.")
        if key in (ord("+"), ord("=")):
            line_x_raw = min(orig_w - 1, line_x_raw + 10)
            print(f"Line x = {line_x_raw}")
        if key == ord("-"):
            line_x_raw = max(1, line_x_raw - 10)
            print(f"Line x = {line_x_raw}")
        if key == ord("s"):
            path = f"frame_{frame_idx}.png"
            cv2.imwrite(path, last_canvas)
            print(f"Saved: {path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="RecycleVision Belt Counter")
    p.add_argument("--video",        required=True)

    # Detector selection
    p.add_argument("--detector",     default="yolo", choices=["yolo", "mog2"],
                   help="Detection backend: 'yolo' (default) or 'mog2'")

    # YOLO-specific
    p.add_argument("--model",        default="best.pt",
                   help="[yolo] Path to .pt weights (default: best.pt; auto-downloads fallback)")
    p.add_argument("--conf",         type=float, default=0.25,
                   help="[yolo] Confidence threshold (default 0.25)")
    p.add_argument("--infer-every",  type=int,   default=2,
                   help="[yolo] Run inference every N frames (default 2)")

    # MOG2-specific
    p.add_argument("--threshold",    type=float, default=40.0,
                   help="[mog2] MOG2 varThreshold — lower = more sensitive (default 40)")
    p.add_argument("--min-area",     type=int,   default=800,
                   help="[mog2] Min blob area px² (default 800)")

    # Shared
    p.add_argument("--speed",        type=float, default=0.5)
    p.add_argument("--line",         type=float, default=0.55,
                   help="Counting line 0.0–1.0 fraction of frame width")
    p.add_argument("--roi",          default=None,
                   help="Restrict detection zone: x1,y1,x2,y2")
    p.add_argument("--save-video",   action="store_true")
    args = p.parse_args()

    run(
        video_path    = args.video,
        detector_type = args.detector,
        model_path    = args.model,
        speed         = args.speed,
        line_pos      = args.line,
        roi           = parse_roi(args.roi),
        conf          = args.conf,
        infer_every   = args.infer_every,
        var_threshold = args.threshold,
        min_area      = args.min_area,
        save_video    = args.save_video,
    )


if __name__ == "__main__":
    main()
