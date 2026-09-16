#!/usr/bin/env python3
"""Figures for the reading deck, in the plain style of the paper figures. All values come
from stored artifacts or the paper tables. Output: docs/study/figs/reading/*.png"""
import os, sys, json, glob
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "paper", "iclr2027", "figs"))
from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
setup(11)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reading")
TAB = os.path.join(ROOT, "paper", "iclr2027", "tables")
def savep(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", pad_inches=0.05, facecolor="white"); plt.close(fig); print("wrote", name)
def rows_of(texfile):
    out = []
    for line in open(os.path.join(TAB, texfile)):
        if "&" in line and "\\\\" in line and not line.strip().startswith(("Model", "Base", "\\multicolumn")):
            out.append([c.strip().replace("\\", "").replace("$", "") for c in line.split("\\\\")[0].split("&")])
    return out
NAME = {"Qwen3-0.6B-Base": "Qwen3-0.6B", "Qwen3-1.7B-Base": "Qwen3-1.7B", "Qwen3-4B-Base": "Qwen3-4B", "Qwen3-8B-Base": "Qwen3-8B",
        "OLMo-2-0425-1B": "OLMo-2-1B", "OLMo-2-1124-7B": "OLMo-2-7B", "Mistral-7B-v0.3": "Mistral-7B", "EuroLLM-1.7B": "EuroLLM-1.7B",
        "bloom-1b7": "BLOOM-1.7B", "bloom-7b1": "BLOOM-7.1B", "SmolLM2-1.7B": "SmolLM2-1.7B", "Falcon3-7B-Base": "Falcon3-7B",
        "salamandra-2b": "Salamandra-2B", "salamandra-7b": "Salamandra-7B", "granite-3.1-8b-base": "granite-8B", "Yi-1.5-9B": "Yi-1.5-9B", "Llama-3.1-8B": "Llama-3.1-8B"}
PARAMS = {"Qwen3-0.6B-Base": 0.6, "Qwen3-1.7B-Base": 1.7, "Qwen3-4B-Base": 4.0, "Qwen3-8B-Base": 8.2, "OLMo-2-0425-1B": 1.0, "OLMo-2-1124-7B": 7.3,
          "Mistral-7B-v0.3": 7.2, "EuroLLM-1.7B": 1.7, "bloom-1b7": 1.7, "bloom-7b1": 7.1, "SmolLM2-1.7B": 1.7, "Falcon3-7B-Base": 7.5,
          "salamandra-2b": 2.3, "salamandra-7b": 7.8, "granite-3.1-8b-base": 8.2, "Yi-1.5-9B": 8.8, "Llama-3.1-8B": 8.0}
MODELS = list(NAME)
GRID = os.path.join(ROOT, "results", "grid")
prof = {}
for m in MODELS:
    d = json.load(open(os.path.join(GRID, m, "metrics.json"))); ks = sorted(d["per_layer"], key=int)
    prof[m] = np.array([d["per_layer"][k]["lfs"]["lfs"] for k in ks])
depth = {m: float(v[0] - v.min()) for m, v in prof.items()}
ci = {}
for m in MODELS:
    p = os.path.join(ROOT, "results", "dip_bootstrap", m, "bootstrap.json")
    if os.path.exists(p): ci[m] = json.load(open(p))["dip_depth"]["q025_median_q975"]

# ---- R1: profile band with Qwen3-8B numbers
xs = np.linspace(0, 1, 201); Y = np.array([np.interp(xs, np.linspace(0, 1, len(v)), v) for v in prof.values()])
fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True); clean(ax, "y")
ax.fill_between(xs, Y.min(0), Y.max(0), color=BAND, lw=0)
ax.plot(xs, np.median(Y, 0), color=BLUE, lw=2.4)
v8 = prof["Qwen3-8B-Base"]; x8 = np.linspace(0, 1, len(v8)); i8 = int(v8.argmin())
ax.plot(x8, v8, color=ORANGE, lw=2.2)
for xi, yi, lab, dx, dy, ha in [(0, v8[0], f"layer 0: {v8[0]:.3f}", 0.02, -0.06, "left"), (x8[i8], v8[i8], f"layer {i8}: {v8[i8]:.3f}", 0.03, -0.02, "left"), (1, v8[-1], f"layer {len(v8)-1}: {v8[-1]:.3f}", -0.02, -0.06, "right")]:
    ax.scatter([xi], [yi], s=46, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=5)
    ax.text(xi + dx, yi + dy, lab, color=ORANGE, fontsize=10.5, ha=ha, va="top")
ax.text(0.56, 0.745, "range of 17 models", color=BLUE, fontsize=10.5, va="center")
ax.text(0.30, np.interp(0.30, xs, np.median(Y, 0)) + 0.025, "median of 17 models", color=BLUE, fontsize=10.5, ha="left", va="bottom")
ax.text(0.52, 0.535, "Qwen3-8B", color=ORANGE, fontsize=10.5, va="top")
ax.axhline(0.298, color=INK2, lw=0.9, ls=(0, (1, 2.5)))
ax.text(0.985, 0.31, "expected value under pure noise, 0.298", color=INK2, fontsize=10, ha="right", va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(0.25, 1.0); ax.set_yticks([0.3, 0.5, 0.7, 0.9]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["input (layer 0)", "middle", "output (last layer)"])
ax.set_ylabel("Language-Factor Share"); savep(fig, "r1_profiles.png")

# ---- R2: dip depth dot plot with intervals and values
order = sorted(MODELS, key=lambda m: depth[m])
fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True); clean(ax, "x")
for i, m in enumerate(order):
    if m in ci:
        ax.plot([ci[m][0], ci[m][2]], [i, i], color=BLUE_L, lw=4, solid_capstyle="butt", zorder=2)
        ax.scatter([ci[m][1]], [i], s=40, color=BLUE, edgecolor="white", linewidth=0.9, zorder=4)
        ax.scatter([depth[m]], [i], s=44, facecolor="white", edgecolor=ORANGE, linewidth=1.2, zorder=5)
    else:
        ax.scatter([depth[m]], [i], s=40, color=BLUE, edgecolor="white", linewidth=0.9, zorder=4)
    ax.text(max(depth[m], ci[m][2] if m in ci else 0) + 0.012, i, f"{depth[m]:.3f}", va="center", fontsize=10, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[m] for m in order], fontsize=10.5); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 0.45); ax.set_ylim(-0.7, len(order) - 0.3); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_xlabel("dip depth (LFS at layer 0 minus its minimum)")
ax.text(0.44, 0.2, "hollow: first-300 grid value (printed); dot: median of 2,000 resamples\nof 300 sentences at the stored layer; bar: 95% interval (14 models)", fontsize=9.5, color=INK2, ha="right", va="bottom")
savep(fig, "r2_dip_depths.png")

# ---- R3: dip depth against parameter count, family series joined
fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True); clean(ax, "both")
series = {"Qwen3": ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base", "Qwen3-8B-Base"], "OLMo-2": ["OLMo-2-0425-1B", "OLMo-2-1124-7B"],
          "BLOOM": ["bloom-1b7", "bloom-7b1"], "Salamandra": ["salamandra-2b", "salamandra-7b"]}
for fam, ms in series.items():
    ax.plot([PARAMS[m] for m in ms], [depth[m] for m in ms], color=GREY_L, lw=2.2, zorder=1)
ax.scatter([PARAMS[m] for m in MODELS], [depth[m] for m in MODELS], s=48, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
off = {"Qwen3-0.6B-Base": (6, -14), "Qwen3-1.7B-Base": (6, 4), "Qwen3-4B-Base": (-8, 6), "Qwen3-8B-Base": (6, 2), "OLMo-2-0425-1B": (6, -4), "OLMo-2-1124-7B": (6, -4),
       "Mistral-7B-v0.3": (-8, 6), "EuroLLM-1.7B": (6, -4), "bloom-1b7": (6, -6), "bloom-7b1": (6, -6), "SmolLM2-1.7B": (6, 3), "Falcon3-7B-Base": (6, -2),
       "salamandra-2b": (6, 3), "salamandra-7b": (6, -13), "granite-3.1-8b-base": (6, 7), "Yi-1.5-9B": (6, -2), "Llama-3.1-8B": (-8, 8)}
for m in MODELS:
    dx, dy = off[m]; ax.annotate(NAME[m], (PARAMS[m], depth[m]), textcoords="offset points", xytext=(dx, dy), fontsize=9.5, color=INK2, ha="left" if dx > 0 else "right")
ax.set_xscale("log"); ax.set_xticks([0.6, 1, 2, 4, 8]); ax.set_xticklabels(["0.6B", "1B", "2B", "4B", "8B"]); ax.set_xlim(0.45, 12)
ax.set_ylim(0, 0.4); ax.set_xlabel("parameters"); ax.set_ylabel("dip depth")
ax.text(0.02, 0.97, "grey lines join sizes within one family", transform=ax.transAxes, fontsize=10, color=INK2, va="top")
savep(fig, "r3_size_vs_family.png")

# ---- R4: hub test, languages won by the multilingual average
wins = {}
for f in sorted(glob.glob(os.path.join(GRID, "*", "metrics.json"))):
    m = os.path.basename(os.path.dirname(f))
    if m not in NAME: continue
    hub = json.load(open(f)).get("hub_at_best_layer") or {}
    if hub:
        adv = [r["latent_advantage"] for r in hub.values() if isinstance(r, dict)]
        wins[m] = (sum(1 for a in adv if a > 0), len(adv), float(np.median(adv)))
ms = sorted(wins, key=lambda k: wins[k][0])
fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True); clean(ax, "x")
for i, m in enumerate(ms):
    w, n, med = wins[m]; ax.plot([100, w], [i, i], color=GREY_L, lw=1.2, zorder=1); ax.scatter([w], [i], s=40, color=BLUE, edgecolor="white", linewidth=0.9, zorder=3)
    ax.text(w + 0.6, i, f"{w} of {n}   (median gain +{med:.2f} R²)", va="center", fontsize=9.5, color=INK2)
ax.set_yticks(range(len(ms))); ax.set_yticklabels([NAME[m] for m in ms], fontsize=10.5); ax.tick_params(axis="y", length=0)
ax.set_xlim(100, 140); ax.set_xticks([100, 110, 120, 127]); ax.set_ylim(-0.7, len(ms) - 0.3)
ax.set_xlabel("languages, of 127, better predicted from the average of the other languages than from English")
savep(fig, "r4_hub_wins.png")

# ---- R5: two behaviors, same measures
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True, constrained_layout=True)
rows = ["1 − LFS-VC", "MEXA (reference measure)", "AaR@10 (tail)"]
left = [(0.041, -0.05, 0.12), (0.238, 0.18, 0.28), (0.105, 0.05, 0.14)]; right = [(0.508, 0.19, 0.71), (0.591, 0.33, 0.82), (0.443, 0.09, 0.71)]
for a_, vals, title in [(ax, left, "(a) pooled benchmark accuracy, 33 models"), (bx, right, "(b) benefit of foreign-language context, 19 models")]:
    clean(a_, "x"); a_.axvline(0, color=INK2, lw=0.8)
    for i, (v, lo, hi) in enumerate(vals):
        y = len(vals) - 1 - i; c = ORANGE if i == 0 else BLUE
        a_.plot([lo, hi], [y, y], color=BLUE_L if i else "#f6c4b0", lw=5, solid_capstyle="butt", zorder=2); a_.scatter([v], [y], s=48, color=c, edgecolor="white", linewidth=0.9, zorder=3)
        a_.text(hi + 0.02, y, f"{v:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=9.5, color=INK2)
    a_.set_xlim(-0.2, 1.15); a_.set_xticks([0, 0.25, 0.5, 0.75]); a_.set_title(title); a_.set_xlabel("Spearman after covariate adjustment")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows[::-1], fontsize=10.5); ax.tick_params(axis="y", length=0); bx.tick_params(axis="y", length=0)
savep(fig, "r5_two_behaviors.png")

# ---- R6: the foreign-context test, as a flat diagram
fig, ax = plt.subplots(figsize=(11, 4.9)); ax.set_xlim(0, 11); ax.set_ylim(-0.6, 4.4); ax.axis("off")
def card(x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - 0.22, body, ha="center", va="center", fontsize=10, color=INK2)
card(0.2, 2.35, 3.3, 1.85, "Matched context", "sentence 1 of a news story, in German\n(or any of 41 languages)\nfollowed by sentence 2 in English", "#e3efe6", "#2e7d5b")
card(0.2, 0.2, 3.3, 1.85, "Mismatched context", "sentence 1 from a different story,\nsame language, same length,\nfollowed by the same sentence 2", "#fbe4e0", "#c0392b")
card(4.2, 1.25, 2.9, 1.9, "Score", "the model's loss on the\ntrue words of sentence 2\nunder each context", "#fff3d6", "#b8860b")
card(7.8, 1.25, 3.0, 1.9, "Benefit of the context", "loss(mismatched) − loss(matched),\ndivided by the same quantity\nwith English context", BAND, BLUE)
for (x1, y1, x2, y2) in [(3.5, 3.1, 4.2, 2.5), (3.5, 1.1, 4.2, 1.85), (7.1, 2.2, 7.8, 2.2)]:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK2))
ax.text(5.5, -0.3, "100 sentence pairs, 41 context languages, 19 models, 157,700 scored sequences; protocol frozen before the run", ha="center", fontsize=10, color=INK2)
savep(fig, "r6_context_test.png")

# ---- R7: evidence 2 as one two-panel figure at slide size (hub wins + offset-plus-scale)
setup(9.5)
import statistics as st
rows = []
for p in glob.glob(os.path.join(ROOT, "results", "budget", "*_fixed_raw", "budget.json")):
    d = json.load(open(p)); pl = d["per_language"]
    rows.append((NAME[d["model"]], st.mean(v["share_M1"]["mean"] for v in pl.values()), st.mean(v["share_M2"]["mean"] for v in pl.values())))
rows.sort(key=lambda r: r[1] + r[2])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 4.9), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.05]})
clean(ax, "x"); ms = sorted(wins, key=lambda k: wins[k][0])
for i, m in enumerate(ms):
    w, n, med = wins[m]; ax.plot([105, w], [i, i], color=GREY_L, lw=1.2, zorder=1); ax.scatter([w], [i], s=30, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(w + 0.5, i, f"{w}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(ms))); ax.set_yticklabels([NAME[m] for m in ms], fontsize=8.5); ax.tick_params(axis="y", length=0)
ax.set_xlim(105, 130); ax.set_xticks([110, 120, 127]); ax.set_ylim(-0.7, len(ms) - 0.3)
ax.set_xlabel("languages of 127 better predicted by the\naverage of other languages than by English", fontsize=9); ax.set_title("(a) is the shared middle English? No", fontsize=10)
clean(bx, "x")
for i, (n, o, s) in enumerate(rows):
    bx.barh(i, o, height=0.66, color=BLUE, zorder=3); bx.barh(i, s, left=o, height=0.66, color=BLUE_L, zorder=3)
    bx.text(o + s + 0.012, i, f"{100 * (o + s):.0f}%", va="center", fontsize=8.5, color=INK2)
bx.set_yticks(range(len(rows))); bx.set_yticklabels([r[0] for r in rows], fontsize=8.5); bx.tick_params(axis="y", length=0)
bx.set_xlim(0, 1.0); bx.set_ylim(-0.6, len(rows) - 0.4); bx.set_xticks([0, 0.5, 1.0]); bx.set_xticklabels(["0", "50%", "100%"])
bx.set_xlabel("held-out misalignment removed by\nper-language offset (dark) plus scale (light)", fontsize=9); bx.set_title("(b) is the map simple? Mostly an offset", fontsize=10)
savep(fig, "r7_evidence2.png")
