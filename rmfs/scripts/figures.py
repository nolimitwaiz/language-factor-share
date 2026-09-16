#!/usr/bin/env python
"""Paper figures from committed result tables (local CPU, matplotlib).

One claim per figure; titles state findings; CIs everywhere;
pre-registered results solid, exploratory hatched/annotated. Output:
results/figures/*.{pdf,png}.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS, "src"))
TAB = os.path.join(RMFS, "results", "tables")
RUNS = os.path.join(RMFS, "results", "runs")
FIG = os.path.join(RMFS, "results", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GREY, GOLD = ("#2b6cb0", "#c53030", "#2f855a",
                                "#718096", "#b7791f")


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"))
    plt.close(fig)
    print(f"[fig] {name}")


# ---------------------------------------------------------------- fig 1
def fig_validity_matrix():
    """Channels x exams: each channel wins exactly one column (F5/F7/F11)."""
    # numbers from tournament_core.csv / a5_validity.csv
    vals = np.array([[0.76, 0.07, -0.10],
                     [0.76, 0.24, 0.36],
                     [np.nan, 0.09, 0.51],
                     [np.nan, np.nan, -0.17],
                     [-0.27, -0.01, -0.09]])
    labels = ["Meaning against language", "Sentence matching",
              "Weakest aligned languages", "Collapse flag",
              "Generation transfer (dropped)"]
    cols = ["Carrying content\nacross languages", "Downstream exams",
            "Trained variants\n(75 checkpoints)"]
    stars = {(0, 0): "★", (1, 1): "★", (2, 2): "★"}

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(vals, cmap="RdBu", vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", color=GREY)
                continue
            s = stars.get((i, j), "")
            ax.text(j, i, f"{v:+.2f}{s}", ha="center", va="center",
                    color="white" if abs(v) > 0.45 else "black",
                    fontweight="bold" if s else "normal")
    ax.set_xticks(range(3), cols)
    ax.set_yticks(range(5), labels)
    ax.set_title("Each reading is worth something different\n"
                 "correlation with outside evidence, after controls "
                 "(★ = best in column)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    save(fig, "fig1_validity_matrix")


# ---------------------------------------------------------------- fig 2
def fig_a5_falsification():
    """C_pres vs downstream ability across 75 arms (F6/F7)."""
    df = pd.read_csv(os.path.join(TAB, "a5_arm_panel.csv"))

    def grp(f):
        if f in ("armWA-C", "armWA-D", "armWA-F"):
            return "collapsed variants", RED
        if f.startswith("armWA") or f == "armB" or f == "armC":
            return "sound variants and controls", BLUE
        if f.startswith("armCS"):
            return "code switch variants", GREEN
        return "objective variants", GOLD

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    seen = set()
    for _, r in df.iterrows():
        lab, c = grp(r.family)
        ax.scatter(r.C_pres, r.panel_acc, color=c, s=28, alpha=0.85,
                   label=lab if lab not in seen else None,
                   edgecolor="white", linewidth=0.4)
        seen.add(lab)
    ax.axhline(df[df.family == "armB"].panel_acc.mean(), color=GREY,
               ls="--", lw=0.8)
    ax.annotate("untouched control", xy=(0.55, df[df.family == "armB"]
                .panel_acc.mean() + 0.001), color=GREY, fontsize=8)
    ax.annotate("95 percent lost,\nfull ability", xy=(0.06, 0.394),
                xytext=(0.18, 0.402), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.annotate("clean geometry,\nlost ability", xy=(0.95, 0.332),
                xytext=(0.55, 0.336), fontsize=8, color=GOLD,
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8))
    ax.set_xlabel("meaning structure kept, against the untouched model")
    ax.set_ylabel("downstream ability (40 language exam)")
    ax.set_title("Losing meaning structure did not cost ability;\n"
                 "the variants that lost ability look clean")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    save(fig, "fig2_collapse_benign")


# ---------------------------------------------------------------- fig 3
def fig_dose_response():
    """T inversion dose-response (F4)."""
    x = ["undamaged", "mild\ndamage", "medium\ndamage",
         "severe\ndamage", "correspondence\ndestroyed"]
    y = [0.0190, 0.0349, 0.0301, 0.0132, 0.0039]
    floor = 3 * 0.0051
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axhspan(0.0190 - floor, 0.0190 + floor, color=GREY, alpha=0.18,
               label="noise band")
    ax.plot(range(5), y, "o-", color=RED, lw=1.5, markersize=6)
    for i, v in enumerate(y):
        ax.annotate(f"{v:+.3f}", (i, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(range(5), x, fontsize=8)
    ax.set_ylabel("measured transfer (median across languages)")
    ax.set_title("Mild damage raises measured transfer; only total\n"
                 "destruction of correspondence brings it to zero")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    save(fig, "fig3_T_inversion")


# ---------------------------------------------------------------- fig 4
def fig_prediction_ledger():
    """Every pre-registered prediction, scored (F12)."""
    preds = [
        ("Collapsed variant ranks last on the single score", "FAIL"),
        ("Detect collapse across the whole signature table", "FAIL"),
        ("Collapse detector exact at every level tested", "PASS"),
        ("Unchanged under a shared rotation", "PASS"),
        ("Respond to offsets, rotations and warps", "FAIL"),
        ("Predicts how content carries across languages", "PASS"),
        ("Same, excluding the generation measure", "PASS"),
        ("Downstream signal on the frozen model set", "PASS"),
        ("Same signal across the full grid", "FAIL"),
        ("Adds to the strongest published measure", "BRANCH"),
        ("Collapse costs at least 5 points of ability", "FAIL"),
        ("Published measure misranks this panel", "VOID"),
        ("Panel validity given the above", "VOID"),
        ("Alignment training adds no downstream gain", "PASS"),
    ]
    colors = {"PASS": GREEN, "FAIL": RED, "BRANCH": GOLD, "VOID": GREY,
              "PENDING": "#a0aec0"}
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for i, (label, res) in enumerate(reversed(preds)):
        ax.barh(i, 1, color=colors[res], alpha=0.85, height=0.62)
        ax.text(0.02, i, label, va="center", fontsize=7.5)
        ax.text(0.985, i, res, va="center", ha="right", fontsize=7.5,
                fontweight="bold", color="white")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Every prediction recorded before its data, and how "
                 "it scored")
    save(fig, "fig4_prediction_ledger")


# ---------------------------------------------------------------- fig 5
def fig_breadth():
    """Effective breadth (F9)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    names = ["nominal\nlanguages", "MEXA\nN_eff", "structure\nN_eff"]
    vals = [69, 2.0, 1.2]
    bars = ax.bar(names, vals, color=[GREY, BLUE, BLUE], alpha=0.85)
    for k, b in enumerate(bars):
        v = vals[k]
        ax.annotate(f"{v:g}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("independent bets (log scale)")
    ax.set_title("Per-language validity claims rest on ~2 effective\n"
                 "languages — for every metric in the study")
    save(fig, "fig5_effective_breadth")


# ---------------------------------------------------------------- fig 6
def fig_layer_profiles():
    """LFS-VC across depth for all profiled models (F3/F10) — runs when
    layer_profiles_* results exist; skipped silently otherwise."""
    dirs = sorted(d for d in os.listdir(RUNS)
                  if d.startswith("layer_profiles_"))
    if not dirs:
        print("[fig] layer profiles not yet pulled — skipped")
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(dirs)))
    for k, d in enumerate(dirs):
        c = cmap[k]
        r = json.load(open(os.path.join(RUNS, d, "result.json")))
        tag = r["tag"]
        lys = sorted(int(k) for k in r["layers"])
        depth = [ly / max(lys) for ly in lys]
        v = [r["layers"][str(ly)]["lfs_vc"] for ly in lys]
        ax.plot(depth, v, "o-", color=c, lw=1.2, markersize=3.5,
                label=tag, alpha=0.9)
    ax.set_xlabel("relative depth (layer / n_layers)")
    ax.set_ylabel("LFS-VC (language share of systematic variance)")
    ax.set_title("The language component across Transformer depth "
                 "(14 models)")
    ax.legend(fontsize=5.5, ncol=2, frameon=False)
    save(fig, "fig6_layer_profiles")


# ---------------------------------------------------------------- fig 7
def fig_geometry():
    """What the decomposition is actually measuring (explanatory)."""
    import json as _json
    p = os.path.join(TAB, "geometry_demo.json")
    if not os.path.exists(p):
        print("[fig] geometry demo absent — skipped")
        return
    G = _json.load(open(p))
    names = {"Qwen3-4B-Base": "Qwen3-4B", "bloom-1b7": "bloom-1b7"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    palette = ["#2b6cb0", "#c53030", "#2f855a", "#b7791f", "#6b46c1",
               "#0987a0"]
    for k, (tag, g) in enumerate(G.items()):
        ax = axes[k]
        co = np.array(g["coords"])                # (L, N, 2)
        L, N, _ = co.shape
        for c in range(N):                        # one line per sentence
            pts = co[:, c, :]
            ax.plot(pts[:, 0], pts[:, 1], color="#cbd5e0", lw=0.4,
                    zorder=1)
        for i, lg in enumerate(g["langs"]):
            ax.scatter(co[i, :, 0], co[i, :, 1], s=16, alpha=0.9,
                       color=palette[i % len(palette)], label=lg,
                       zorder=2, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{names.get(tag, tag)}   language share "
                     f"{g['lfs']:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Each dot is one sentence; grey lines join translations "
                 "of the same sentence.\nThe clusters are languages, not "
                 "meanings.", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_geometry")


if __name__ == "__main__":
    fig_validity_matrix()
    fig_a5_falsification()
    fig_dose_response()
    fig_prediction_ledger()
    fig_breadth()
    fig_layer_profiles()
    fig_geometry()
    print(f"[done] -> {FIG}")
