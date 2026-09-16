#!/usr/bin/env python3
"""Appendix figure: five training conditions on Qwen3-0.6B read as the LFS ratio and as the raw content variance.
Reads results/aim2/evaluation_wordalign.json."""
import json, os
import numpy as np
from figstyle import *
setup(8.5)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ev = json.load(open(os.path.join(ROOT, "results", "aim2", "evaluation_wordalign.json"))); SEEDS = ["0", "1", "2"]
def mean_over_seeds(arm, key, layer="8"): return float(np.mean([ev["seeds"][s][arm]["trained"][layer][key] for s in SEEDS if arm in ev["seeds"].get(s, {})]))
arms = [("WA-A", "frozen checkpoint"), ("WA-B", "LM-only control"), ("WA-C", "word-alignment MSE"), ("WA-E", "segment MSE"), ("WA-G", "cosine")]
lfs = [mean_over_seeds(a, "lfs") for a, _ in arms]; raw = [mean_over_seeds(a, "v_concept_raw") for a, _ in arms]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.0), sharey=True, constrained_layout=True)
ys = list(range(len(arms)))[::-1]
for (a, n), y, v, w in zip(arms, ys, lfs, raw):
    c = ORANGE if a == "WA-C" else BLUE
    ax.scatter([v], [y], s=42, color=c, edgecolor="white", linewidth=0.9, zorder=3); ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    bx.scatter([w], [y], s=42, color=c, edgecolor="white", linewidth=0.9, zorder=3); bx.text(w * 1.15, y, f"{w:,.0f}", va="center", fontsize=7.5, color=INK2)
for a_ in (ax, bx): clean(a_, "x"); a_.tick_params(axis="y", length=0)
ax.set_yticks(ys); ax.set_yticklabels([n for _, n in arms], fontsize=7.5)
ax.set_xlim(0.42, 0.53); ax.set_xticks([0.44, 0.48, 0.52]); ax.set_xlabel("LFS"); ax.set_title("(a) the ratio barely moves")
bx.set_xscale("log"); bx.set_xlim(400, 60000); bx.set_xlabel("raw content variance (log scale)"); bx.set_title("(b) the component falls twenty-fold")
save(fig, "fig17_ratio_hides_scale")
