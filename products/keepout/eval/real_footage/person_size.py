"""How small can a person be before YOLOX-tiny (416 px input) stops finding them,
with one full-frame pass versus full frame plus 2x2 overlapping tiles?

Truth: people found at full size (single pass, score >= 0.35) in real frames of the
Malta and courtyard clips. Each frame is then shrunk by a factor and pasted onto a
960x540 grey canvas, so the frame stays the same size and the people get smaller
relative to it. Recall is measured per person, binned by the person's height as a
fraction of frame height, for both detection modes. Plus the time per frame of each.
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from visioncore import YoloxDetector, iter_video

from keepout.track import iou

HERE = Path(__file__).resolve().parent
REAL = Path(os.environ.get("KEEPOUT_REAL_FOOTAGE", HERE))
CLIPS = [REAL / "clips/malta-mdina-shot4-74.35s-104.6s.mp4",
         REAL / "clips/courtyard-vtest-40s.mp4"]
W, H = 960, 540
det = YoloxDetector(score_threshold=0.35)


def people(img):
    return [d.bbox for d in det.detect(img) if d.class_id == 0]


def tiles(img, overlap=0.15):
    h, w = img.shape[:2]
    tw, th = round(w * (0.5 + overlap)), round(h * (0.5 + overlap))
    out = list(people(img))
    for y0 in (0, h - th):
        for x0 in (0, w - tw):
            out += [(b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0)
                    for b in people(img[y0:y0 + th, x0:x0 + tw])]
    return out


def canvas_of(frame, s):
    fh, fw = frame.shape[:2]
    base = min(W / fw, H / fh)
    small = cv2.resize(frame, (round(fw * base * s), round(fh * base * s)),
                       interpolation=cv2.INTER_AREA)
    c = np.full((H, W, 3), 110, np.uint8)
    ox, oy = (W - small.shape[1]) // 2, (H - small.shape[0]) // 2
    c[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    return c, base * s, ox, oy


bins = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.0]
stats = {m: np.zeros((len(bins) - 1, 2), int) for m in ("single", "tiled")}
ms = {"single": [], "tiled": []}
for clip in CLIPS:
    for fr in iter_video(clip, stride=25, max_frames=24):
        base_img, k, ox, oy = canvas_of(fr.image, 1.0)
        truth = [b for b in people(base_img)]
        truth = [((b[0] - ox) / k, (b[1] - oy) / k, (b[2] - ox) / k, (b[3] - oy) / k)
                 for b in truth]
        for s in (1.0, 0.75, 0.55, 0.4, 0.3, 0.22):
            img, k, ox, oy = canvas_of(fr.image, s)
            exp = [(b[0] * k + ox, b[1] * k + oy, b[2] * k + ox, b[3] * k + oy) for b in truth]
            for mode, fn in (("single", people), ("tiled", tiles)):
                t0 = time.perf_counter()
                found = fn(img)
                ms[mode].append((time.perf_counter() - t0) * 1000)
                for e in exp:
                    frac = (e[3] - e[1]) / H
                    bi = np.searchsorted(bins, frac, side="right") - 1
                    hit = any(iou(e, f) >= 0.3 for f in found)
                    stats[mode][bi] += (int(hit), 1)
rows = []
for i in range(len(bins) - 1):
    row = {"height_fraction": f"{bins[i]:.2f}-{bins[i+1]:.2f}"}
    for mode in stats:
        hit, n = stats[mode][i]
        row[mode] = None if n == 0 else round(hit / n, 3)
        row[f"{mode}_n"] = int(n)
    rows.append(row)
out = {"rows": rows, "ms_per_frame": {m: round(float(np.median(v)), 1) for m, v in ms.items()}}
print(json.dumps(out, indent=1))
(HERE / "person-size.json").write_text(json.dumps(out, indent=1))
