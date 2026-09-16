#!/usr/bin/env python3
"""Figure 2. Sentence-component share at the dip layer against median R_content, 19 models.
Reads results/round3/r1_content_transfer_analysis_2026-08-24/model_level_summary.csv."""
import csv, os, math
from figstyle import *
setup(8.5)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "round3", "r1_content_transfer_analysis_2026-08-24", "model_level_summary.csv"))))
for r in rows: r["x"] = float(r["q_L"]); r["y"] = float(r["median_R_content"])
short = {"Falcon3-3B-Base": "Falcon3-3B", "Llama-3.1-8B": "Llama-3.1-8B", "Mistral-Nemo-Base-2407": "Mistral-Nemo-12B",
         "OLMo-2-1124-13B": "OLMo-2-13B", "Qwen2.5-0.5B": "Qwen2.5-0.5B", "Qwen2.5-1.5B": "Qwen2.5-1.5B",
         "Qwen2.5-3B": "Qwen2.5-3B", "Qwen2.5-7B": "Qwen2.5-7B", "Sailor-1.8B": "Sailor-1.8B", "Sailor-7B": "Sailor-7B",
         "SmolLM2-360M": "SmolLM2-360M", "TowerBase-7B": "Tower-7B", "Yi-1.5-9B": "Yi-1.5-9B",
         "granite-3.1-8b-base": "granite-8B", "mGPT": "mGPT", "occiglot-7b-eu5": "occiglot-7B",
         "xglm-1.7B": "XGLM-1.7B", "xglm-2.9B": "XGLM-2.9B", "xglm-7.5B": "XGLM-7.5B"}
XL, XH, YL, YH = 0.05, 0.72, 0.18, 0.76; W, H = 4.8, 3.4
fig, ax = plt.subplots(figsize=(W, H), constrained_layout=True); clean(ax, "both")
ax.set_xlim(XL, XH); ax.set_ylim(YL, YH)
ax.scatter([r["x"] for r in rows], [r["y"] for r in rows], s=42, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
def to_ax(x, y): return (x - XL) / (XH - XL), (y - YL) / (YH - YL)
FS = 7.2; cw = 0.56 * FS / 72 / (W * 0.85); ch = 1.3 * FS / 72 / (H * 0.85)
pts = [to_ax(r["x"], r["y"]) for r in rows]; boxes = []; placed = {}
def overlaps(b, others): return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in others)
def pt_in(b, pts_, pad=0.014): return any(b[0] - pad < px < b[2] + pad and b[1] - pad < py < b[3] + pad for px, py in pts_)
for i in sorted(range(len(rows)), key=lambda i: (pts[i][1], pts[i][0])):
    r = rows[i]; px, py = pts[i]; lab = short[r["model"]]; w = len(lab) * cw; h = ch; best = None
    for dist in (0.025, 0.05, 0.08, 0.12):
        for ang in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
            a = math.radians(ang); cx, cy = px + dist * math.cos(a), py + dist * math.sin(a) * 1.15
            x0 = cx - w / 2 if ang in (90, -90) else (cx if abs(ang) < 90 else cx - w); y0 = cy - h / 2; b = (x0, y0, x0 + w, y0 + h)
            if b[0] < 0.005 or b[2] > 0.995 or b[1] < 0.005 or b[3] > 0.92: continue
            if overlaps(b, boxes) or pt_in(b, [p for j, p in enumerate(pts) if j != i]): continue
            best = (b, dist, ang); break
        if best: break
    if best is None: best = ((px + 0.03, py - h / 2, px + 0.03 + w, py + h / 2), 0.03, 0)
    boxes.append(best[0]); placed[r["model"]] = best
for r in rows:
    b, dist, ang = placed[r["model"]]; px, py = to_ax(r["x"], r["y"])
    ha, tx = ("left", b[0]) if abs(ang) < 90 else (("center", (b[0] + b[2]) / 2) if abs(ang) == 90 else ("right", b[2])); ty = (b[1] + b[3]) / 2
    kw = dict(fontsize=FS, color=INK2, ha=ha, va="center", zorder=5)
    if dist > 0.06: ax.annotate(short[r["model"]], (px, py), xytext=(tx, ty), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, shrinkA=0, shrinkB=3), **kw)
    else: ax.text(tx, ty, short[r["model"]], transform=ax.transAxes, **kw)
ax.set_xlabel("sentence-component share at the dip layer (1 − LFS-VC)")
ax.set_ylabel("median benefit of foreign-language context ($R_{\\mathrm{content}}$)")
ax.text(0.02, 0.97, "Spearman 0.77 [0.44, 0.94], 19 models", transform=ax.transAxes, fontsize=8, color=INK, va="top")
save(fig, "fig_r1_model_level")
