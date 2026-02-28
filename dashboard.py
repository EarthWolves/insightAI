"""
InsightAI — Construction Site Intelligence Dashboard
Overlays Grounding DINO detection with live construction analytics on a video feed.

Usage:
    python dashboard.py --video timelapse.mp4
    python dashboard.py --video timelapse.mp4 --speed 0.25 --save-video

Controls:
    q / ESC  — quit
    SPACE    — pause / resume
    r        — reset DINO counters
"""

import argparse
import random
import time
from collections import defaultdict, deque
import cv2
import numpy as np

from detect import format_classes, load_model
from camera import InferenceWorker

# ── Canvas layout ──────────────────────────────────────────────────────────────
CW, CH  = 1600, 900
HDR_H   = 52
G       = 8

BODY_Y  = HDR_H + G           # 60
BODY_H  = CH - BODY_Y - G     # 832

LW      = 754                  # left col width
VH      = LW * 360 // 640     # video height (424)
RX      = G + LW + G          # right col x (770)
RW      = CW - RX - G         # right col width (822)

PH_H    = 202                  # phase panel height
EV_H    = 300                  # events panel height
MT_H    = 188                  # materials panel height
IN_H    = BODY_H - PH_H - EV_H - MT_H - 3 * G  # insights panel height (118)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
BG      = ( 17,  21,  29)
PNL     = ( 25,  29,  41)
HDR_BG  = ( 10,  13,  21)
BDR     = ( 50,  57,  76)
SEP     = ( 38,  44,  60)

WHT     = (228, 234, 244)
GRY     = (126, 134, 152)
DIM     = ( 65,  71,  90)

GRN     = ( 65, 198,  90)    # green
AMB     = ( 28, 162, 255)    # amber
RED_C   = ( 52,  62, 228)    # red
BLU     = (208, 148,  72)    # blue accent
CYN     = (185, 208,  60)    # cyan/teal
PRP     = (190,  95, 172)    # purple

F0      = cv2.FONT_HERSHEY_SIMPLEX
F1      = cv2.FONT_HERSHEY_DUPLEX

# ── Construction plan ─────────────────────────────────────────────────────────
PHASES = [
    # (name,                        pct,   color)
    ("Site Prep & Excavation",      100.0, GRN),
    ("Foundation & Concrete",       100.0, GRN),
    ("Structural Steel Framing",     85.0, CYN),
    ("MEP Rough-In",                 42.0, AMB),
    ("Exterior Envelope",            23.0, AMB),
    ("Interior Finishing",            6.0, RED_C),
    ("Landscaping & Final",           0.0, DIM),
]

MATERIALS = [
    # (name,              in_today,   on_site,    required,  status,     color)
    ("Structural Steel",  "24.2 T",   "187.4 T",  "340 T",   "ON TRACK", GRN),
    ("Rebar",             "2.1 T",    "8.3 T",    "22.0 T",  "LOW",      AMB),
    ("Concrete",          "0 loads",  "18.4 m3",  "~80 m3",  "PENDING",  CYN),
    ("Formwork Panels",   "0 units",  "535 stg",  "900",     "OK",       GRN),
    ("Safety Barriers",   "+12",      "145 dep",  "200",     "OK",       GRN),
]

# (frame_trigger, severity, sim_time, actor, message)
EVENTS_DATA = [
    (150,  "WARN", "08:14", "Truck 657BGD3",       "Arrived 32min late -> Rescheduling Level 4 steel drop to 13:45"),
    (320,  "INFO", "09:07", "Forklift FL-04",      "Idle 23min detected -> Redeployed to NE quadrant staging"),
    (480,  "WARN", "09:45", "Rebar Inventory",     "Short delivery: 48 bundles rcv'd (exp. 52) -> Expediting Supplier B"),
    (640,  "CRIT", "10:32", "Crane OP-7",          "Wind hold activated 11:45 — Level 5+ operations suspended"),
    (780,  "INFO", "11:12", "Crane OP-7",          "Wind hold cleared 12:10 -> Resuming Level 5 beam placement"),
    (950,  "INFO", "12:05", "Concrete C-089",      "4 loads confirmed 14:00 pour — Pump C-112 replacement done"),
    (1120, "INFO", "12:48", "Crew B (12 workers)", "Early arrival 07:45 -> Pre-staged 84 forms for Level 4 deck"),
    (1300, "WARN", "13:22", "Material Move",       "Rebar R-441: South staging -> NE corner (Level 4 prep)"),
    (1500, "INFO", "13:55", "Phase 3C Milestone",  "Level 4 West steel complete — 1.2 days ahead of schedule"),
    (1680, "INFO", "14:18", "Truck 441ZXB9",       "Departing — 4.1T scrap steel (8 beams). Manifest verified"),
    (1860, "INFO", "14:47", "Safety Sweep",        "Zone C cleared — 12 barriers repositioned for Level 5 access"),
    (2050, "CRIT", "15:03", "Weather Alert",       "28mph wind at 16:00 -> Pre-positioning crane for shutdown 15:30"),
    (2240, "INFO", "15:29", "Concrete Pour C-4",   "Level 4 East deck complete — 18.4 m3 placed. Cure: 28 hrs"),
    (2480, "INFO", "16:01", "Phase Progress",      "Overall 61.4% complete — On track for Oct 15, 2026 completion"),
    (2690, "WARN", "16:38", "Overtime Alert",      "Crew A approaching 10hr limit -> Shift handoff to Crew C at 16:45"),
]

INSIGHTS = [
    ("Stage rebar in NE quadrant",     "Level 4 pour starts in ~2.5 hrs — pre-position now"),
    ("Crane utilization: 67% (v13%)",  "Reschedule beam lifts to AM slots to hit 80% target"),
    ("Phase 3 ahead of plan +3 days",  "Projected completion Dec 18 — buffer for weather risk"),
    ("Peak truck window: 10:00-13:00", "4.2 deliveries/hr avg — schedule bulk loads then"),
    ("FL-04 & FL-06 route overlap",    "FL-06 redirected to Zone D -> efficiency +12%"),
    ("Concrete: pre-order 24h ahead",  "Supplier B avg response time 4.2 hrs — plan accordingly"),
    ("Level 5 steel window: 13-14h",   "Clear before wind advisory — proceed with beam placement"),
    ("Phase 4 MEP pre-install ready",  "Level 3 West structural signed off — crews can start"),
]

CLASSES = "excavator, bulldozer, backhoe loader, dump truck, crane, worker"


# ── State ──────────────────────────────────────────────────────────────────────

class SiteState:
    def __init__(self):
        self.event_log: deque = deque(maxlen=9)
        self._triggered: set  = set()
        self.dino_counts: dict = defaultdict(int)
        # live phase percentages — start from plan baseline, drift upward
        self.phase_pcts: list = [p[1] for p in PHASES]
        self._last_phase_tick = 0

    def update(self, frame_idx: int, new_dets: list | None):
        for i, (trigger, sev, ts, actor, msg) in enumerate(EVENTS_DATA):
            if i not in self._triggered and frame_idx >= trigger:
                self._triggered.add(i)
                self.event_log.appendleft((ts, sev, actor, msg))
        if new_dets:
            for d in new_dets:
                self.dino_counts[d["label"]] += 1
        # nudge phase progress every 30 frames — later phases move slower
        if frame_idx - self._last_phase_tick >= 30:
            self._last_phase_tick = frame_idx
            max_inc = [0, 0, 0.45, 0.28, 0.18, 0.10, 0.04]
            for i, inc in enumerate(max_inc):
                if self.phase_pcts[i] < 100 and inc > 0:
                    self.phase_pcts[i] = min(100.0, self.phase_pcts[i] + random.uniform(0, inc))


# ── Low-level draw helpers ─────────────────────────────────────────────────────

def txt(c, s, x, y, color=WHT, scale=0.50, thick=1, font=F0):
    cv2.putText(c, str(s), (x, y), font, scale, color, thick, cv2.LINE_AA)

def panel_bg(c, x, y, w, h, title="", tc=CYN):
    cv2.rectangle(c, (x, y), (x + w, y + h), PNL, -1)
    cv2.rectangle(c, (x, y), (x + w, y + h), BDR,  1)
    if title:
        txt(c, title, x + 10, y + 20, tc, 0.46, 1)
        cv2.line(c, (x, y + 27), (x + w, y + 27), SEP, 1)

def hbar(c, x, y, w, h, pct, color, bg=DIM):
    cv2.rectangle(c, (x, y), (x + w, y + h), bg, -1)
    fw = max(0, int(w * min(pct, 100) / 100))
    if fw:
        cv2.rectangle(c, (x, y), (x + fw, y + h), color, -1)

def sev_clr(sev):
    return {"INFO": BLU, "WARN": AMB, "CRIT": RED_C}.get(sev, GRY)

def sim_time(frame_idx, total):
    """Map video progress to a simulated 08:00–17:00 workday."""
    mins = int(frame_idx / max(total, 1) * 540)
    return f"{8 + mins // 60:02d}:{mins % 60:02d}"


# ── Panel renderers ────────────────────────────────────────────────────────────

def render_header(c, frame_idx, total):
    cv2.rectangle(c, (0, 0), (CW, HDR_H), HDR_BG, -1)
    cv2.line(c, (0, HDR_H), (CW, HDR_H), BDR, 1)

    txt(c, "INSIGHTAI", 14, 34, CYN, 0.80, 2, F1)
    txt(c, "CONSTRUCTION SITE INTELLIGENCE  //  Huntington Apartments Phase II  //  Apex Construction LLC", 135, 24, WHT, 0.47, 1)
    txt(c, "Budget: $18.2M    Target: Oct 15 2026    85 units / 6 stories    GC: Apex Construction LLC", 135, 41, GRY, 0.38, 1)

    # video progress bar top-right
    bx, by, bw = CW - 310, 12, 240
    pct = frame_idx / max(total, 1)
    cv2.rectangle(c, (bx, by), (bx + bw, by + 10), DIM, -1)
    cv2.rectangle(c, (bx, by), (bx + int(bw * pct), by + 10), GRN, -1)
    txt(c, f"VIDEO {pct*100:.0f}%  |  SIM TIME {sim_time(frame_idx, total)}", bx, by + 26, GRY, 0.42, 1)


def render_video(c, frame, fps):
    vf = cv2.resize(frame, (LW, VH))
    c[BODY_Y: BODY_Y + VH, G: G + LW] = vf
    cv2.rectangle(c, (G, BODY_Y), (G + LW, BODY_Y + VH), BDR, 1)
    # live badge
    cv2.rectangle(c, (G + 6, BODY_Y + 6), (G + 88, BODY_Y + 22), GRN, -1)
    txt(c, f"LIVE  {fps:.0f} FPS", G + 9, BODY_Y + 18, BG, 0.46, 1)


def render_counters(c, state: SiteState):
    cy = BODY_Y + VH + G
    ch = CH - cy - G
    cv2.rectangle(c, (G, cy), (G + LW, cy + ch), PNL, -1)
    cv2.rectangle(c, (G, cy), (G + LW, cy + ch), BDR, 1)
    txt(c, "DETECTED ASSETS  (Grounding DINO)", G + 10, cy + 20, CYN, 0.47, 1)
    cv2.line(c, (G, cy + 28), (G + LW, cy + 28), SEP, 1)

    display_classes = ["excavator", "bulldozer", "backhoe loader", "dump truck", "crane", "worker"]
    ncols = 3
    cw = LW // ncols
    rh = (ch - 32) // 2

    for i, cls in enumerate(display_classes):
        col, row = i % ncols, i // ncols
        cx_ = G + col * cw + cw // 2
        ry_ = cy + 32 + row * rh

        count = state.dino_counts.get(cls, 0)
        clr   = CYN if count > 0 else DIM

        cs    = str(count)
        (nw, nh), _ = cv2.getTextSize(cs, F1, 1.8, 3)
        txt(c, cs, cx_ - nw // 2, ry_ + nh + 8, clr, 1.8, 3, F1)
        (lw, _), _ = cv2.getTextSize(cls.upper(), F0, 0.42, 1)
        txt(c, cls.upper(), cx_ - lw // 2, ry_ + nh + 26, GRY, 0.42, 1)

    # grid lines
    for col in range(1, ncols):
        cv2.line(c, (G + col * cw, cy + 29), (G + col * cw, CH - G), SEP, 1)
    cv2.line(c, (G, cy + 32 + rh), (G + LW, cy + 32 + rh), SEP, 1)

    total = sum(state.dino_counts.values())
    txt(c, f"Session total detections: {total}", G + 10, CH - G - 5, DIM, 0.40, 1)


def render_phases(c, frame_idx, state: "SiteState"):
    x, y, w, h = RX, BODY_Y, RW, PH_H
    panel_bg(c, x, y, w, h, "PHASE PROGRESS  //  Huntington Apartments Phase II")

    overall = sum(state.phase_pcts) / len(state.phase_pcts)
    txt(c, "OVERALL", x + 10, y + 44, GRY, 0.40, 1)
    txt(c, f"{overall:.1f}%", x + 68, y + 44, GRN, 0.56, 2)
    txt(c, "Budget: $18.2M     Start: Jan 06 2025     Target: Oct 15 2026", x + 130, y + 44, GRY, 0.38, 1)
    hbar(c, x + 10, y + 50, w - 20, 7, overall, GRN)

    name_w = 190
    bar_x  = x + name_w + 10
    bar_w  = w - name_w - 70

    for i, (name, _, clr) in enumerate(PHASES):
        pct  = state.phase_pcts[i]
        ry   = y + 65 + i * 19
        fclr = WHT if pct > 0 else DIM
        txt(c, name, x + 10, ry + 11, fclr, 0.41, 1)
        hbar(c, bar_x, ry, bar_w, 12, pct, clr)
        txt(c, f"{pct:.1f}%", bar_x + bar_w + 6, ry + 11, clr if pct > 0 else DIM, 0.42, 1)


def render_events(c, state: SiteState):
    x, y, w, h = RX, BODY_Y + PH_H + G, RW, EV_H
    panel_bg(c, x, y, w, h, "REAL-TIME EVENTS  //  Live Site Feed")

    if not state.event_log:
        txt(c, "Waiting for events...", x + 12, y + 52, DIM, 0.46, 1)
        return

    events = list(state.event_log)
    row_h  = max(26, (h - 34) // min(len(events), 9))

    for i, (ts, sev, actor, msg) in enumerate(events[:9]):
        ry = y + 33 + i * row_h
        if ry + row_h > y + h - 2:
            break
        sc = sev_clr(sev)

        # severity pill
        (sw, _), _ = cv2.getTextSize(sev, F0, 0.33, 1)
        pw = sw + 8
        cv2.rectangle(c, (x + 8, ry + 2), (x + 8 + pw, ry + 15), sc, -1)
        txt(c, sev, x + 12, ry + 13, BG, 0.33, 1)

        txt(c, ts,    x + 18 + pw, ry + 13, GRY,  0.41, 1)
        txt(c, actor, x + 68 + pw, ry + 13, sc,   0.43, 1)

        # message on second line — truncate to fit panel width
        max_chars = (w - 20) // 7
        display_msg = msg if len(msg) <= max_chars else msg[:max_chars - 1] + "…"
        txt(c, display_msg, x + 14, ry + 25, WHT, 0.40, 1)

        if i < len(events) - 1:
            cv2.line(c, (x + 8, ry + row_h - 1), (x + w - 8, ry + row_h - 1), SEP, 1)


def render_materials(c):
    x, y, w, h = RX, BODY_Y + PH_H + G + EV_H + G, RW, MT_H
    panel_bg(c, x, y, w, h, "MATERIAL INVENTORY  //  Today")

    headers = [(x + 10, "MATERIAL"), (x + 190, "IN TODAY"), (x + 308, "ON-SITE"),
               (x + 430, "REQUIRED"), (x + 560, "STATUS")]
    for hx, hl in headers:
        txt(c, hl, hx, y + 43, GRY, 0.39, 1)
    cv2.line(c, (x + 6, y + 47), (x + w - 6, y + 47), SEP, 1)

    row_h = (h - 54) // len(MATERIALS)
    for i, (name, in_t, on_s, req, status, sc) in enumerate(MATERIALS):
        ry = y + 54 + i * row_h
        if i % 2 == 0:
            cv2.rectangle(c, (x + 4, ry - 1), (x + w - 4, ry + row_h - 1), (30, 35, 48), -1)
        txt(c, name,   x + 10,  ry + 13, WHT, 0.43, 1)
        txt(c, in_t,   x + 190, ry + 13, WHT, 0.43, 1)
        txt(c, on_s,   x + 308, ry + 13, WHT, 0.43, 1)
        txt(c, req,    x + 430, ry + 13, GRY, 0.43, 1)
        (bw, _), _ = cv2.getTextSize(status, F0, 0.37, 1)
        cv2.rectangle(c, (x + 556, ry + 3), (x + 556 + bw + 10, ry + 17), sc, -1)
        txt(c, status, x + 561, ry + 14, BG, 0.37, 1)


def render_insights(c, frame_idx):
    x, y, w, h = RX, BODY_Y + PH_H + G + EV_H + G + MT_H + G, RW, IN_H
    panel_bg(c, x, y, w, h, "AI OPTIMIZATION  //  InsightAI Engine v2.1")

    n    = len(INSIGHTS)
    base = (frame_idx // 220) % n
    vis  = [INSIGHTS[(base + i) % n] for i in range(3)]
    rh   = max(28, (h - 32) // 3)

    for i, (title, detail) in enumerate(vis):
        ry = y + 32 + i * rh
        cv2.circle(c, (x + 18, ry + 8), 4, CYN, -1)
        txt(c, title,  x + 30, ry + 12, CYN, 0.46, 1)
        txt(c, detail, x + 30, ry + 26, GRY, 0.39, 1)


# ── Column divider ────────────────────────────────────────────────────────────

def render_divider(c):
    dx = G + LW + G // 2
    cv2.line(c, (dx, BODY_Y), (dx, CH - G), BDR, 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_dashboard(
    video_path:  str,
    classes:     str   = CLASSES,
    threshold:   float = 0.30,
    speed:       float = 0.25,
    model_id:    str   = "IDEA-Research/grounding-dino-tiny",
    infer_every: int   = 3,
    save_video:  bool  = False,
):
    processor, model, device = load_model(model_id)
    text_prompt = format_classes(classes)
    print(f"\nDetecting: {text_prompt}")
    print("Controls:  q / ESC = quit   SPACE = pause   r = reset counters\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open '{video_path}'")
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay_ms     = max(1, int(1000 / (src_fps * speed)))

    print(f"Video: {int(cap.get(3))}x{int(cap.get(4))} @ {src_fps:.0f}fps  |  "
          f"{total_frames} frames  |  playback {speed}x  ({delay_ms}ms/frame)\n")

    worker = InferenceWorker(processor, model, device, text_prompt, threshold)
    worker.start()

    writer = None
    if save_video:
        out_path = video_path.rsplit(".", 1)[0] + "_dashboard.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                   src_fps * speed, (CW, CH))
        print(f"Saving dashboard video to: {out_path}")

    state        = SiteState()
    frame_idx    = 0
    paused       = False
    fps          = 0.0
    t_prev       = time.perf_counter()
    last_canvas  = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"\nEnd of video.  Final DINO counts: {dict(state.dino_counts)}")
                break

            if frame_idx % infer_every == 0:
                worker.submit(frame)

            new_dets = worker.pop_detections()
            state.update(frame_idx, new_dets)

            t_now  = time.perf_counter()
            fps    = 1.0 / max(t_now - t_prev, 1e-9)
            t_prev = t_now

            # Build composite canvas
            canvas = np.full((CH, CW, 3), BG, dtype=np.uint8)
            render_header(canvas, frame_idx, total_frames)
            render_video(canvas, frame, fps)
            render_counters(canvas, state)
            render_divider(canvas)
            render_phases(canvas, frame_idx, state)
            render_events(canvas, state)
            render_materials(canvas)
            render_insights(canvas, frame_idx)

            last_canvas = canvas
            if writer:
                writer.write(canvas)

            frame_idx += 1

        if last_canvas is not None:
            cv2.imshow("InsightAI — Construction Dashboard", last_canvas)

        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            print("Paused." if paused else "Resumed.")
        if key == ord("r"):
            state.dino_counts.clear()
            print("DINO counters reset.")

    worker.stop()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="InsightAI Construction Dashboard")
    p.add_argument("--video",       required=True)
    p.add_argument("--classes",     default=CLASSES)
    p.add_argument("--threshold",   type=float, default=0.30)
    p.add_argument("--speed",       type=float, default=0.25,
                   help="Playback speed multiplier (default 0.25 = 4x slower)")
    p.add_argument("--model",       default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--infer-every", type=int, default=3)
    p.add_argument("--save-video",  action="store_true")
    args = p.parse_args()

    run_dashboard(
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
