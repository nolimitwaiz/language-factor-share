#!/usr/bin/env python3
"""Figure 1. (a) LFS by relative depth: band = range of the 17 direct-SS models, line = median,
one model highlighted. (b) dip depth per model with the 95% sentence-subsampling interval.
Reads results/grid/<m>/metrics.json and results/dip_bootstrap/<m>/bootstrap.json."""
import json, os
import numpy as np
from figstyle import *
setup(8.5)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GRID_DIR = os.path.join(ROOT, "results", "grid"); BOOT = os.path.join(ROOT, "results", "dip_bootstrap")
MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base", "Qwen3-8B-Base", "OLMo-2-0425-1B",
          "OLMo-2-1124-7B", "Mistral-7B-v0.3", "EuroLLM-1.7B", "bloom-1b7", "bloom-7b1", "SmolLM2-1.7B",
          "Falcon3-7B-Base", "salamandra-2b", "salamandra-7b", "granite-3.1-8b-base", "Yi-1.5-9B", "Llama-3.1-8B"]
NAME = {"Qwen3-0.6B-Base": "Qwen3-0.6B", "Qwen3-1.7B-Base": "Qwen3-1.7B", "Qwen3-4B-Base": "Qwen3-4B",
        "Qwen3-8B-Base": "Qwen3-8B", "OLMo-2-0425-1B": "OLMo-2-1B", "OLMo-2-1124-7B": "OLMo-2-7B",
        "Mistral-7B-v0.3": "Mistral-7B", "EuroLLM-1.7B": "EuroLLM-1.7B", "bloom-1b7": "BLOOM-1.7B",
        "bloom-7b1": "BLOOM-7.1B", "SmolLM2-1.7B": "SmolLM2-1.7B", "Falcon3-7B-Base": "Falcon3-7B",
        "salamandra-2b": "Salamandra-2B", "salamandra-7b": "Salamandra-7B", "granite-3.1-8b-base": "granite-8B",
        "Yi-1.5-9B": "Yi-1.5-9B", "Llama-3.1-8B": "Llama-3.1-8B"}
xs = np.linspace(0, 1, 201); Y = []; prof = {}; depth = {}
for m in MODELS:
    d = json.load(open(os.path.join(GRID_DIR, m, "metrics.json")))
    ks = sorted(d["per_layer"], key=int); v = np.array([d["per_layer"][k]["lfs"]["lfs"] for k in ks])
    x = np.linspace(0, 1, len(v)); prof[m] = (x, v); depth[m] = float(v[0] - v.min()); Y.append(np.interp(xs, x, v))
Y = np.array(Y)
ci = {m: json.load(open(os.path.join(BOOT, m, "bootstrap.json")))["dip_depth"]["q025_median_q975"]
      for m in MODELS if os.path.exists(os.path.join(BOOT, m, "bootstrap.json"))}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.75), gridspec_kw={"width_ratios": [1.2, 1.0]}, constrained_layout=True)
clean(ax, "y"); clean(bx, "x")
ax.fill_between(xs, Y.min(0), Y.max(0), color=BAND, lw=0, zorder=1)
ax.plot(xs, np.median(Y, 0), color=BLUE, lw=2.0, zorder=3)
x8, v8 = prof["Qwen3-8B-Base"]; ax.plot(x8, v8, color=ORANGE, lw=1.6, zorder=4)
i8 = int(v8.argmin()); ax.scatter([x8[i8]], [v8[i8]], s=26, color=ORANGE, edgecolor="white", linewidth=0.9, zorder=5)
ax.axhline(0.298, color=INK2, lw=0.8, ls=(0, (1, 2.5)), zorder=2)
ax.text(0.56, 0.735, "range of 17 models", color=BLUE, fontsize=7.5, va="center", zorder=6)
ax.text(0.99, np.median(Y, 0)[-1] - 0.045, "median", color=BLUE, fontsize=7.5, ha="right", va="top", zorder=6)
ax.text(x8[i8] + 0.05, v8[i8] - 0.01, "Qwen3-8B,\ndeepest dip", color=ORANGE, fontsize=7.5, va="top", zorder=6)
ax.text(0.99, 0.31, "expected value under pure noise", color=INK2, fontsize=7, ha="right", va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(0.25, 1.0); ax.set_yticks([0.3, 0.5, 0.7, 0.9]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["input", "middle", "output"])
ax.set_ylabel("Language-Factor Share"); ax.set_title("(a) every profile has an interior minimum")

order = sorted(MODELS, key=lambda m: depth[m])
for i, m in enumerate(order):
    if m in ci:
        bx.plot([ci[m][0], ci[m][2]], [i, i], color=BLUE_L, lw=3, solid_capstyle="butt", zorder=2)
        bx.scatter([ci[m][1]], [i], s=24, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)          # resampling median
        bx.scatter([depth[m]], [i], s=26, facecolor="white", edgecolor=ORANGE, linewidth=1.1, zorder=5)   # first-300 grid value
    else:
        bx.scatter([depth[m]], [i], s=24, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
from matplotlib.lines import Line2D
bx.legend(handles=[Line2D([], [], marker="o", color=BLUE, markeredgecolor="white", markersize=5, lw=0, label="resampling median"),
                   Line2D([], [], color=BLUE_L, lw=3, label="95% interval"),
                   Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.1, markersize=5.5, lw=0, label="first-300 grid value")],
          loc="lower right", frameon=False, fontsize=7, handlelength=1.3, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.3)
bx.set_yticks(range(len(order))); bx.set_yticklabels([NAME[m] for m in order], fontsize=7); bx.tick_params(axis="y", length=0)
bx.set_xlim(0, 0.4); bx.set_ylim(-0.7, len(order) - 0.3); bx.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
bx.set_xlabel("dip depth (nine-fold range)"); bx.set_title("(b) dip depth, with 95% resampling intervals")
save(fig, "fig1_lfs_profiles_17")
