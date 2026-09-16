#!/usr/bin/env python3
"""Appendix figure: how much of the held-out cross-language misalignment an offset and a scale remove, 14 models.
Reads results/budget/<m>_fixed_raw/budget.json."""
import json, glob, os, statistics as st
from figstyle import *
setup(8.5)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NAME = {"EuroLLM-1.7B": "EuroLLM-1.7B", "Falcon3-7B-Base": "Falcon3-7B", "Mistral-7B-v0.3": "Mistral-7B", "OLMo-2-0425-1B": "OLMo-2-1B",
        "OLMo-2-1124-7B": "OLMo-2-7B", "Qwen3-0.6B-Base": "Qwen3-0.6B", "Qwen3-1.7B-Base": "Qwen3-1.7B", "Qwen3-4B-Base": "Qwen3-4B",
        "Qwen3-8B-Base": "Qwen3-8B", "SmolLM2-1.7B": "SmolLM2-1.7B", "bloom-1b7": "BLOOM-1.7B", "bloom-7b1": "BLOOM-7.1B",
        "salamandra-2b": "Salamandra-2B", "salamandra-7b": "Salamandra-7B"}
rows = []
for p in glob.glob(os.path.join(ROOT, "results", "budget", "*_fixed_raw", "budget.json")):
    d = json.load(open(p)); pl = d["per_language"]
    rows.append((NAME[d["model"]], st.mean(v["share_M1"]["mean"] for v in pl.values()), st.mean(v["share_M2"]["mean"] for v in pl.values())))
rows.sort(key=lambda r: r[1] + r[2])
fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True); clean(ax, "x")
for i, (n, o, s) in enumerate(rows):
    ax.barh(i, o, height=0.66, color=BLUE, zorder=3); ax.barh(i, s, left=o, height=0.66, color=BLUE_L, zorder=3)
    ax.text(o + s + 0.012, i, f"{100 * (o + s):.0f}%", va="center", fontsize=7.5, color=INK2, zorder=4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of cross-language misalignment removed on held-out sentences")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="per-language offset"), Patch(color=BLUE_L, label="plus a scale")], loc="lower right", handlelength=1.1, borderaxespad=0.6)
save(fig, "fig15_budget_composition")
