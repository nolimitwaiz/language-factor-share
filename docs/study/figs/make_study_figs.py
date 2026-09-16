#!/usr/bin/env python3
"""Schematic figures for the LFS study guide. Synthetic or from saved repo JSON only."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BLUE, ORANGE, AQUA, YELLOW, MAG, INK, INK2, GRID, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.linewidth": 0.7, "xtick.color": INK2, "ytick.color": INK2})
rng = np.random.default_rng(7)

def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)

def box(ax, x, y, w, h, text, fc="#f4f6fa", ec=INK2, fs=8.5, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=color, lw=lw))

# ---------------- 1. gradient descent ----------------
def fig_gd():
    fig, ax = plt.subplots(figsize=(5.2, 3.0)); clean(ax)
    th = np.linspace(-3, 3, 200); L = (th - 1) ** 2 + 0.5
    ax.plot(th, L, color=GRAY, lw=2)
    t = -2.6; pts = [t]
    for _ in range(7):
        t = t - 0.3 * 2 * (t - 1); pts.append(t)
    pts = np.array(pts)
    ax.plot(pts, (pts - 1) ** 2 + 0.5, "o-", color=BLUE, lw=1.4, ms=5)
    for i in range(3):
        ax.annotate("", (pts[i + 1], (pts[i + 1] - 1) ** 2 + 0.5), (pts[i], (pts[i] - 1) ** 2 + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
    ax.text(-2.55, 13.3, "start", fontsize=8, color=INK2)
    ax.text(1.05, 0.9, "minimum: gradient = 0", fontsize=8, color=INK2)
    ax.set_xlabel(r"parameter $\theta$"); ax.set_ylabel(r"loss $L(\theta)$")
    ax.set_title("Gradient descent: step against the slope, shrink as the slope flattens", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_ml_gd.png", dpi=180); plt.close(fig)

# ---------------- 2. linear regression ----------------
def fig_linreg():
    fig, ax = plt.subplots(figsize=(5.2, 3.0)); clean(ax)
    x = np.linspace(0, 10, 25); y = 1.2 * x + 2 + rng.normal(0, 1.6, 25)
    A = np.vstack([x, np.ones_like(x)]).T; w, b = np.linalg.lstsq(A, y, rcond=None)[0]
    ax.scatter(x, y, s=22, color=BLUE, zorder=3)
    ax.plot(x, w * x + b, color=ORANGE, lw=2)
    for xi, yi in zip(x[::3], y[::3]):
        ax.plot([xi, xi], [yi, w * xi + b], color=GRAY, lw=1)
    ax.set_xlabel("input x"); ax.set_ylabel("target y")
    ax.set_title("Least squares: the line that minimizes the sum of squared residuals (gray)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_ml_linreg.png", dpi=180); plt.close(fig)

# ---------------- 3. logistic / softmax ----------------
def fig_logistic():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.8)); clean(ax); clean(bx)
    z = np.linspace(-6, 6, 200); ax.plot(z, 1 / (1 + np.exp(-z)), color=BLUE, lw=2)
    ax.axhline(0.5, color=GRAY, lw=0.8); ax.set_xlabel("logit z"); ax.set_ylabel(r"$\sigma(z)$")
    ax.set_title("Sigmoid: logit to probability", fontsize=9, loc="left")
    logits = np.array([2.0, 0.5, -1.0, 0.0, 1.2]); p = np.exp(logits) / np.exp(logits).sum()
    bx.bar(range(5), p, color=BLUE, width=0.6)
    bx.set_xticks(range(5)); bx.set_xticklabels(["the", "a", "cat", "dog", "is"])
    for i, v in enumerate(p): bx.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    bx.set_ylabel("probability"); bx.set_title("Softmax over a 5-word vocabulary", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_ml_logistic.png", dpi=180); plt.close(fig)

# ---------------- 4. overfitting ----------------
def fig_overfit():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.8)); clean(ax); clean(bx)
    x = np.sort(rng.uniform(0, 1, 14)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 14)
    xs = np.linspace(0, 1, 200)
    for deg, c, lab in ((1, GRAY, "degree 1 (underfit)"), (3, BLUE, "degree 3"), (12, ORANGE, "degree 12 (overfit)")):
        co = np.polyfit(x, y, deg); ax.plot(xs, np.clip(np.polyval(co, xs), -2.5, 2.5), color=c, lw=1.8, label=lab)
    ax.scatter(x, y, s=20, color=INK, zorder=3); ax.set_ylim(-2.2, 2.2); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Same data, three model sizes", fontsize=9, loc="left")
    degs = np.arange(1, 13); tr = 0.9 * np.exp(-0.45 * degs) + 0.05; va = 0.9 * np.exp(-0.45 * degs) + 0.05 + 0.004 * (degs - 3) ** 2 * (degs > 3)
    bx.plot(degs, tr, "o-", color=BLUE, ms=4, label="training error"); bx.plot(degs, va, "o-", color=ORANGE, ms=4, label="held-out error")
    bx.set_xlabel("model capacity (degree)"); bx.legend(fontsize=7, frameon=False)
    bx.set_title("Held-out error is the only honest score", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_ml_overfit.png", dpi=180); plt.close(fig)

# ---------------- 5. MLP ----------------
def fig_mlp():
    fig, ax = plt.subplots(figsize=(5.6, 2.9)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    layers = [(1.2, 3, "input\n$x \\in \\mathbb{R}^3$"), (4.2, 5, "hidden\n$h=\\phi(W_1x+b_1)$"), (7.2, 2, "output\n$\\hat y = W_2 h + b_2$")]
    coords = []
    for cx, n, lab in layers:
        ys = np.linspace(1.2, 4.8, n) if n > 1 else [3.0]
        coords.append([(cx, y) for y in ys])
        for y in ys: ax.add_patch(Circle((cx, y), 0.28, fc="#f4f6fa", ec=INK2, lw=1))
        ax.text(cx, 5.55, lab, ha="center", fontsize=8, color=INK)
    for a, b in ((0, 1), (1, 2)):
        for (x0, y0) in coords[a]:
            for (x1, y1) in coords[b]:
                ax.plot([x0 + 0.28, x1 - 0.28], [y0, y1], color=GRAY, lw=0.6, zorder=0)
    ax.text(4.2, 0.45, "every arrow is a weight; $\\phi$ is a nonlinearity (ReLU, GELU)", ha="center", fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig("fig_ml_mlp.png", dpi=180); plt.close(fig)

# ---------------- 6. LLM pipeline ----------------
def fig_llm_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    steps = [("text\n\"Der Hund\nschläft\"", "#fff4e6"), ("tokens\n[Der][ Hund]\n[ schl][äft]", "#fff4e6"),
             ("embeddings\n$T\\times D$", "#eef4fc"), ("block 1\n...\nblock $n$", "#eef4fc"),
             ("hidden states\n$x_0, x_1, \\dots, x_n$", "#eef4fc"), ("unembed\n+ softmax", "#eafaf2"), ("next-token\nprobabilities", "#eafaf2")]
    w, gap, x = 1.22, 0.2, 0.15
    for text, fc in steps:
        box(ax, x, 0.8, w, 1.5, text, fc=fc, fs=7.6); x += w + gap
        if x < 9.8: arrow(ax, x - gap + 0.02, 1.55, x - 0.02, 1.55)
    ax.text(5.0, 0.35, "LFS reads the hidden states; the model is never asked to generate", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(); fig.savefig("fig_llm_pipeline.png", dpi=180); plt.close(fig)

# ---------------- 7. transformer block / residual stream ----------------
def fig_llm_block():
    fig, ax = plt.subplots(figsize=(6.6, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    # residual stream horizontal line
    ax.plot([0.4, 9.6], [3, 3], color=INK, lw=2.2); ax.text(0.4, 3.35, r"residual stream $x_{k-1}$", fontsize=8, color=INK)
    ax.text(9.55, 3.35, r"$x_k$", fontsize=9, color=INK, ha="right")
    # attention branch
    box(ax, 1.6, 4.2, 1.5, 0.9, "norm", fs=8); box(ax, 3.4, 4.2, 1.9, 0.9, "attention\n(mix across positions)", fs=7.5)
    arrow(ax, 1.9, 3.02, 1.9, 4.18); arrow(ax, 3.12, 4.65, 3.38, 4.65); arrow(ax, 5.32, 4.65, 5.75, 4.65); arrow(ax, 5.75, 4.65, 5.75, 3.05)
    ax.add_patch(Circle((5.75, 3), 0.16, fc="white", ec=INK, lw=1.4)); ax.text(5.75, 3, "+", ha="center", va="center", fontsize=10)
    # mlp branch
    box(ax, 6.3, 1.1, 1.3, 0.9, "norm", fs=8); box(ax, 7.9, 1.1, 1.5, 0.9, "MLP\n(per position)", fs=7.5)
    arrow(ax, 6.6, 2.98, 6.6, 2.02); arrow(ax, 7.62, 1.55, 7.88, 1.55); arrow(ax, 9.42, 1.55, 9.42, 2.95)
    ax.add_patch(Circle((9.42, 3), 0.16, fc="white", ec=INK, lw=1.4)); ax.text(9.42, 3, "+", ha="center", va="center", fontsize=10)
    ax.text(5.0, 0.35, "one block: $x \\leftarrow x + \\mathrm{Attn}(\\mathrm{norm}(x))$, then $x \\leftarrow x + \\mathrm{MLP}(\\mathrm{norm}(x))$; the hidden state at layer $k$ is $x_k$", ha="center", fontsize=8.2, color=INK2)
    fig.tight_layout(); fig.savefig("fig_llm_block.png", dpi=180); plt.close(fig)

# ---------------- 8. attention heatmap ----------------
def fig_attention():
    toks = ["The", "bat", "flew", "through", "the", "air"]
    n = len(toks); A = np.zeros((n, n))
    for i in range(n):
        s = rng.uniform(0.2, 1.0, i + 1); s[0] += 0.4
        if i >= 1: s[1] += 0.9 if toks[i] in ("flew", "air") else 0
        A[i, :i + 1] = s / s.sum()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=35, ha="right"); ax.set_yticks(range(n)); ax.set_yticklabels(toks)
    ax.set_xlabel("attends to (key)"); ax.set_ylabel("query position")
    for i in range(n):
        for j in range(i + 1): ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if A[i, j] > 0.5 else INK)
    ax.set_title("Causal attention: each row is a distribution over earlier tokens", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_llm_attention.png", dpi=180); plt.close(fig)

# ---------------- 9. LFS grid ----------------
def fig_lfs_grid():
    fig, ax = plt.subplots(figsize=(7.0, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    langs = ["eng", "deu", "hin", "swa"]; sents = ["s1", "s2", "s3", "s4", "s5"]
    x0, y0, cw, ch = 1.6, 1.4, 1.15, 0.9
    for i, l in enumerate(langs):
        for j, s in enumerate(sents):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (3 - i) * ch), cw, ch, fc="#eef4fc", ec=INK2, lw=0.8))
            ax.text(x0 + j * cw + cw / 2, y0 + (3 - i) * ch + ch / 2, r"$\mathbf{z}_{%s,%s}$" % (l, s), ha="center", va="center", fontsize=7.5)
        ax.text(x0 - 0.15, y0 + (3 - i) * ch + ch / 2, l, ha="right", va="center", fontsize=9, color=INK)
        ax.add_patch(Rectangle((x0 + 5 * cw + 0.25, y0 + (3 - i) * ch), 1.5, ch, fc="#fff4e6", ec=ORANGE, lw=1))
        ax.text(x0 + 5 * cw + 1.0, y0 + (3 - i) * ch + ch / 2, r"$\bar{\mathbf{z}}_{%s}$" % l, ha="center", va="center", fontsize=8.5)
    for j, s in enumerate(sents):
        ax.text(x0 + j * cw + cw / 2, y0 + 4 * ch + 0.15, s, ha="center", fontsize=9, color=INK)
        ax.add_patch(Rectangle((x0 + j * cw, y0 - 1.15), cw, 0.9, fc="#eafaf2", ec=AQUA, lw=1))
        ax.text(x0 + j * cw + cw / 2, y0 - 0.7, r"$\bar{\mathbf{z}}_{%s}$" % s, ha="center", va="center", fontsize=8.5)
    ax.text(x0 + 5 * cw + 1.0, y0 + 4 * ch + 0.15, "language means", ha="center", fontsize=8.5, color=ORANGE)
    ax.text(x0 - 0.15, y0 - 0.7, "sentence\nmeans", ha="right", va="center", fontsize=8.5, color=AQUA)
    ax.text(x0 + 5 * cw + 1.0, y0 - 0.7, r"grand mean $\bar{\mathbf{z}}$", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(9.3, 4.6, r"$\mathrm{SS}_{\mathrm{lang}} = N\sum_l\|\bar{\mathbf{z}}_l-\bar{\mathbf{z}}\|^2$", fontsize=9, color=ORANGE)
    ax.text(9.3, 3.8, r"$\mathrm{SS}_{\mathrm{con}} = L\sum_c\|\bar{\mathbf{z}}_c-\bar{\mathbf{z}}\|^2$", fontsize=9, color=AQUA)
    ax.text(9.3, 2.9, r"$\mathrm{LFS}=\dfrac{\mathrm{SS}_{\mathrm{lang}}}{\mathrm{SS}_{\mathrm{lang}}+\mathrm{SS}_{\mathrm{con}}}$", fontsize=10, color=INK)
    ax.text(9.3, 1.6, "each cell is one $D$-dimensional\nmean-pooled sentence vector,\njointly z-scored per coordinate", fontsize=7.8, color=INK2, va="top")
    ax.set_title("The balanced grid: rows are languages, columns are matched sentences", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig("fig_lfs_grid.png", dpi=180); plt.close(fig)

# ---------------- 10. decomposition bars from a real model ----------------
def fig_lfs_decomp():
    d = json.load(open(os.path.join(ROOT, "results", "grid", "Qwen3-8B-Base", "metrics.json")))
    pl = d["per_layer"]; ks = sorted(pl, key=int)
    vals = [pl[k]["lfs"]["lfs"] for k in ks]; mi = int(np.argmin(vals))
    picks = [(0, "layer 0"), (mi, f"layer {mi} (dip)"), (len(ks) - 1, f"layer {len(ks)-1} (last)")]
    fig, ax = plt.subplots(figsize=(5.6, 2.9)); clean(ax, grid="x")
    for i, (k, lab) in enumerate(picks):
        v = pl[ks[k]]["lfs"]; parts = [v["var_lang"], v["var_concept"], v["var_resid"]]; left = 0
        for part, c, name in zip(parts, (ORANGE, AQUA, GRAY), ("language", "concept", "residual")):
            ax.barh(i, part, left=left, color=c, height=0.55, label=name if i == 0 else None); left += part
        ax.text(1.01, i, f"LFS = {v['lfs']:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(3)); ax.set_yticklabels([p[1] for p in picks]); ax.set_xlim(0, 1.18); ax.set_xlabel("share of total sum of squares")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, ncol=3, bbox_to_anchor=(0.86, -0.42))
    ax.set_title("Qwen3-8B: the same decomposition at three layers (128 languages, 300 sentences)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_lfs_decomp.png", dpi=180); plt.close(fig)

# ---------------- 11. null curves ----------------
def fig_lfs_null():
    fig, ax = plt.subplots(figsize=(5.2, 3.0)); clean(ax)
    N = np.arange(20, 2001)
    for L, c in ((12, GRAY), (41, AQUA), (128, BLUE)):
        ax.plot(N, (L - 1) / (L + N - 2), color=c, lw=2, label=f"L = {L}")
    ax.scatter([300, 1500], [127 / 426, 127 / 1626], color=BLUE, s=30, zorder=3)
    ax.annotate("0.298 at (128, 300)", (300, 127 / 426), xytext=(420, 0.36), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.annotate("0.078 at (128, 1500)", (1500, 127 / 1626), xytext=(1150, 0.16), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel("sentences N"); ax.set_ylabel("pure-noise LFS"); ax.legend(frameon=False, fontsize=8)
    ax.set_title(r"The null $(L-1)/(L+N-2)$ depends on the grid, not on $D$", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_lfs_null.png", dpi=180); plt.close(fig)

# ---------------- 12. two-way ANOVA visual ----------------
def fig_anova():
    fig, ax = plt.subplots(figsize=(5.6, 3.0)); clean(ax)
    langs = ["eng", "deu", "hin", "swa"]; off = np.array([0.0, 0.4, 1.3, 1.9]); con = np.array([0, 2, 4, 6, 8, 10]) * 0.55
    for i, l in enumerate(langs):
        y = off[i] + con + rng.normal(0, 0.18, len(con))
        ax.scatter(con / 0.55, y, s=26, color=[BLUE, ORANGE, AQUA, MAG][i], label=l, zorder=3)
        ax.plot(con / 0.55, off[i] + con, color=[BLUE, ORANGE, AQUA, MAG][i], lw=1, alpha=0.6)
    ax.set_xlabel("sentence (concept) index"); ax.set_ylabel("one hidden coordinate")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.text(5.0, 0.3, "vertical offsets between lines = language effect\nslope along a line = concept effect\nscatter around lines = residual", fontsize=8, color=INK2, va="bottom")
    ax.set_title("Two-way decomposition on one coordinate", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_stats_anova.png", dpi=180); plt.close(fig)

# ---------------- 13. bootstrap by unit ----------------
def fig_bootstrap():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.8)); clean(ax); clean(bx)
    # synthetic: 4 models x 34 languages, model-level effect dominates
    models = np.array([0.2, 0.45, 0.5, 0.7]); rows = []
    for m in models:
        rows.append(np.column_stack([np.full(34, m) + rng.normal(0, 0.03, 34), m + rng.normal(0, 0.05, 34)]))
    X = np.vstack(rows); mid = np.repeat(np.arange(4), 34)
    from scipy.stats import spearmanr
    lang_b, model_b = [], []
    for _ in range(600):
        idx = rng.integers(0, len(X), len(X)); lang_b.append(spearmanr(X[idx, 0], X[idx, 1]).statistic)
        pick = rng.integers(0, 4, 4); idx = np.concatenate([np.where(mid == p)[0] for p in pick]); model_b.append(spearmanr(X[idx, 0], X[idx, 1]).statistic)
    ax.hist(lang_b, bins=30, color=BLUE, alpha=0.9); ax.set_title("resample rows (languages): narrow", fontsize=9, loc="left"); ax.set_xlabel("Spearman")
    bx.hist(np.array(model_b)[~np.isnan(model_b)], bins=30, color=ORANGE, alpha=0.9); bx.set_title("resample models (n = 4): wide", fontsize=9, loc="left"); bx.set_xlabel("Spearman")
    ax.set_xlim(0.5, 1.0); bx.set_xlim(-0.2, 1.05)
    fig.suptitle("Same data, two bootstrap units. The unit you resample is the population you generalize to.", fontsize=9, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig("fig_stats_bootstrap.png", dpi=180); plt.close(fig)

# ---------------- 14. deflation ----------------
def fig_deflation():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.8)); clean(ax); clean(bx)
    n = 120; tokens = rng.normal(0, 1, n)
    metric = 0.8 * tokens + rng.normal(0, 0.6, n); acc = 0.8 * tokens + 0.25 * metric + rng.normal(0, 0.6, n)
    ax.scatter(metric, acc, s=14, color=BLUE); ax.set_xlabel("intrinsic metric"); ax.set_ylabel("task accuracy")
    from scipy.stats import spearmanr
    ax.set_title(f"raw: Spearman {spearmanr(metric, acc).statistic:.2f}", fontsize=9, loc="left")
    def resid(y, x):
        A = np.column_stack([x, np.ones_like(x)]); return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rm, ra = resid(metric, tokens), resid(acc, tokens)
    bx.scatter(rm, ra, s=14, color=ORANGE); bx.set_xlabel("metric, data size removed"); bx.set_ylabel("accuracy, data size removed")
    bx.set_title(f"deflated: Spearman {spearmanr(rm, ra).statistic:.2f}", fontsize=9, loc="left")
    fig.suptitle("Deflation: a shared driver (training-data size) inflates the raw correlation", fontsize=9, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig("fig_stats_deflation.png", dpi=180); plt.close(fig)

# ---------------- 15. collapse invariance (synthetic) ----------------
def fig_collapse():
    L, N, D = 12, 200, 64
    G = rng.normal(0, 1.0, (L, 1, D)); C = rng.normal(0, 2.0, (1, N, D)); E = rng.normal(0, 0.7, (L, N, D))
    Z = G + C + E
    def lfs_raw(X):
        mu = X.mean((0, 1)); ml = X.mean(1); mc = X.mean(0)
        sl = N * ((ml - mu) ** 2).sum(); sc = L * ((mc - mu) ** 2).sum(); return sl / (sl + sc), sc
    ks = [1.0, 0.5, 0.25, 0.1, 0.05]; lf, sc = zip(*[lfs_raw(k * Z) for k in ks])
    fig, ax = plt.subplots(figsize=(5.2, 2.8)); clean(ax)
    ax.plot(ks, lf, "o-", color=BLUE, lw=2, label="LFS (unchanged)")
    ax2 = ax.twinx(); ax2.plot(ks, np.array(sc) / sc[0], "s--", color=ORANGE, lw=1.6, label="raw concept SS / original")
    ax2.spines["top"].set_visible(False); ax2.set_ylabel("raw concept variance (relative)", color=ORANGE)
    ax.set_xlabel("global shrink factor k"); ax.set_ylabel("LFS", color=BLUE); ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_title("Shrink every vector by k: the ratio does not move, the component collapses", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig("fig_lfs_collapse_synthetic.png", dpi=180); plt.close(fig)

# ---------------- 16. LLM training loop ----------------
def fig_training():
    fig, ax = plt.subplots(figsize=(7.0, 2.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    steps = ["batch of token\nsequences", "forward pass:\npredict each\nnext token", "cross-entropy\nloss vs. true\nnext tokens", "backward pass:\ngradients", "optimizer step\n(Adam): update\nweights"]
    w, gap, x = 1.65, 0.28, 0.2
    for i, t in enumerate(steps):
        box(ax, x, 0.7, w, 1.7, t, fc="#eef4fc" if i < 3 else "#fff4e6", fs=7.6); x += w + gap
        if i < 4: arrow(ax, x - gap + 0.02, 1.55, x - 0.02, 1.55)
    ax.annotate("", (0.35, 0.55), (9.3, 0.55), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, connectionstyle="arc3,rad=0.0"))
    ax.text(5.0, 0.2, "repeat for trillions of tokens; the result is the base checkpoint", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(); fig.savefig("fig_llm_training.png", dpi=180); plt.close(fig)

for f in (fig_gd, fig_linreg, fig_logistic, fig_overfit, fig_mlp, fig_llm_pipeline, fig_llm_block, fig_attention,
          fig_lfs_grid, fig_lfs_decomp, fig_lfs_null, fig_anova, fig_bootstrap, fig_deflation, fig_collapse, fig_training):
    f(); print("wrote", f.__name__)

# ---------------- 17. kinds of learning ----------------
def fig_ml_types():
    fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    kinds = [(0.2, "Supervised\nexamples come with answers\n\nMT pairs, labeled images,\nBelebele scoring", "#fff4e6"),
             (3.15, "Unsupervised\nno answers, find structure\n\nclustering, PCA,\nvariance decomposition", "#fff4e6"),
             (6.1, "Self-supervised\nanswers made from the data\n\nnext-token prediction\n= LLM pretraining (base model)", "#eafaf2"),
             (9.05, "Reinforcement\nlearn from rewards\n\nRLHF / DPO post-training\n(instruct model)", "#fff4e6")]
    for x, t, fc in kinds:
        box(ax, x, 2.4, 2.75, 2.6, t, fc=fc, fs=7.4)
        arrow(ax, 6.0, 5.58, x + 1.375, 5.03)
    box(ax, 3.6, 5.6, 4.8, 1.0, "Machine learning\nchoose a function from examples, by minimizing a loss", fc="#eef4fc", fs=8.5)
    box(ax, 2.6, 0.35, 6.8, 1.35, "Measurement (not learning): LFS, MEXA, probes, the logit lens\nexamine what a trained model contains. A measurement used as a loss becomes\npart of training, and Goodhart's law applies (Chapter 5).", fc="#f4f6fa", fs=7.6)
    fig.tight_layout(); fig.savefig("fig_ml_types.png", dpi=180); plt.close(fig)

fig_ml_types(); print("wrote fig_ml_types")
