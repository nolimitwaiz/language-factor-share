"""Schematic: from parallel sentences to the Language-Factor Share (LFS).

Two-row pipeline diagram, drawn with matplotlib primitives only (no data,
no model inference).  Row 1 follows one sentence through the model to a
single mean-pooled hidden-state vector.  Row 2 shows the 128 x 300 grid
of such vectors, joint per-coordinate standardization, the marginal means,
the two sums of squares, the LFS ratio, and the per-layer depth profile.

Outputs
    paper/figs/fig22_pipeline_diagram.png
    docs/study/figs/report/fig22_pipeline_diagram.png
    (same image, copied)

Run:  ~/.hiero_venv/bin/python code/pipeline_diagram.py
"""
from __future__ import annotations

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

ROOT = os.path.expanduser("~/Desktop/multilingual-metrics")
OUT_MAIN = os.path.join(ROOT, "paper", "figs", "fig22_pipeline_diagram.png")
OUT_COPY = os.path.join(ROOT, "docs", "study", "figs", "report", "fig22_pipeline_diagram.png")

# ---------------------------------------------------------------- palette
NAVY = "#1f3b73"
NAVY_LIGHT = "#e4eaf6"
NAVY_MID = "#7d93c4"
TEAL = "#0b7f86"
TEAL_LIGHT = "#dcf1ef"
TEAL_MID = "#7cc4c7"
ORANGE = "#e8731f"
ORANGE_LIGHT = "#fdebd9"
INK = "#1d1d1f"
GREY = "#5f6570"
ARROW = "#4a5a7a"
WHITE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)

# Canvas in tenths of an inch: 132 x 88 units on a 13.2 x 8.8 inch figure.
W, H = 132.0, 88.0
fig = plt.figure(figsize=(13.2, 8.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


# ---------------------------------------------------------------- helpers
def box(x, y, w, h, fc, ec, lw=1.4, r=0.9, z=1, ls="-"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z,
    )
    ax.add_patch(p)
    return p


def text(x, y, s, size=10, color=INK, ha="center", va="center", weight="normal",
         style="normal", rotation=0, z=5):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, rotation=rotation, zorder=z)


def caption(xc, y, s, color, size=10.5):
    return text(xc, y, s, size=size, color=color, weight="bold")


def arrow(p0, p1, color=ARROW, lw=1.5, ms=13, z=4, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z, linestyle=ls,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def poly_arrow(pts, color=ARROW, lw=1.5, ms=13, z=4):
    a = FancyArrowPatch(path=Path(pts), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def line(pts, color=ARROW, lw=1.5, z=4, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="round")


def dots_v(x, y, color=GREY, gap=0.55, s=9):
    ax.scatter([x] * 3, [y - gap, y, y + gap], s=s, color=color, zorder=6, lw=0)


def dots_h(x, y, color=GREY, gap=0.55, s=9):
    ax.scatter([x - gap, x, x + gap], [y] * 3, s=s, color=color, zorder=6, lw=0)


def cell_grid(x0, y0, ncol, nrow, cw, ch, colors, z=6, ec=WHITE):
    """Grid of small squares; colors is a 2-D array-like of hex strings."""
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(Rectangle((x0 + j * cw, y0 + i * ch), cw, ch,
                                   fc=colors[i][j], ec=ec, lw=0.5, zorder=z))


rng = np.random.default_rng(7)


def shade(base_rgb, n, lo=0.35, hi=1.0):
    """n colours between white-ish and base colour."""
    base = np.array(matplotlib.colors.to_rgb(base_rgb))
    out = []
    for t in rng.uniform(lo, hi, size=n):
        c = 1 - t * (1 - base)
        out.append(matplotlib.colors.to_hex(c))
    return out


# ---------------------------------------------------------------- title & bands
text(W / 2, 85.6, "From parallel sentences to the Language-Factor Share",
     size=15.5, color=NAVY, weight="bold")

# row bands
box(0.6, 53.5, 2.0, 29.0, NAVY, NAVY, lw=0, r=0.5)
text(1.6, 68.0, "1   Inside the model", size=10.5, color=WHITE, weight="bold",
     rotation=90)
box(0.6, 6.0, 2.0, 34.0, TEAL, TEAL, lw=0, r=0.5)
text(1.6, 23.0, "2   The grid and the decomposition", size=10.5, color=WHITE,
     weight="bold", rotation=90)

# ================================================================ ROW 1
YC = 68.0

# --- A. matched translations -------------------------------------------
box(4, 55, 25, 26, NAVY_LIGHT, NAVY)
caption(16.5, 79.2, "Matched translations", NAVY)
text(16.5, 77.4, "same sentence, 128 languages", size=10, color=GREY)

cards = [
    ("EN", ["The bat flew", "through the air."], "normal", INK),
    ("DE", ["Die Fledermaus flog", "durch die Luft."], "normal", INK),
    ("AR", ["Arabic translation", "(right-to-left script)"], "italic", GREY),
]
card_tops = [75.3, 68.7, 62.1]
for (tag, lines, sty, col), top in zip(cards, card_tops):
    cy0 = top - 6.0
    box(5.5, cy0, 22, 6.0, WHITE, NAVY_MID, lw=1.0, r=0.6, z=2)
    # language pill
    box(6.4, cy0 + 1.8, 4.0, 2.4, NAVY, NAVY, lw=0, r=0.5, z=3)
    text(8.4, cy0 + 3.0, tag, size=10, color=WHITE, weight="bold")
    text(11.4, cy0 + 3.95, lines[0], size=10, color=col, ha="left", style=sty)
    text(11.4, cy0 + 2.05, lines[1], size=10, color=col, ha="left", style=sty)

arrow((29.2, YC), (31.8, YC))

# --- B. tokenizer ------------------------------------------------------
box(32, 62.5, 10, 11, NAVY_LIGHT, NAVY)
caption(37, 71.6, "Tokenizer", NAVY)
text(37, 68.7, "sub-word", size=10, color=INK)
text(37, 67.0, "pieces", size=10, color=INK)
pw = [1.2, 1.8, 1.0, 1.5, 1.3]
xx = 33.5
for w_ in pw:
    ax.add_patch(Rectangle((xx, 63.6), w_, 1.4, fc=WHITE, ec=NAVY, lw=0.9, zorder=6))
    xx += w_ + 0.35

arrow((42.2, YC), (44.8, YC))

# --- C. token embeddings ----------------------------------------------
box(45, 60.5, 13, 15, NAVY_LIGHT, NAVY)
caption(51.5, 73.6, "Token", NAVY)
caption(51.5, 71.9, "embeddings", NAVY)
emb_cols = [shade(NAVY, 4) for _ in range(5)]
emb_colors = [[emb_cols[j][i] for j in range(5)] for i in range(4)]
cell_grid(47.6, 64.0, 5, 4, 1.55, 1.25, emb_colors)
text(51.5, 62.2, "T token vectors", size=10, color=GREY)

# into the bottom of the stack
poly_arrow([(58.2, YC), (59.6, YC), (59.6, 55.9), (61.8, 55.9)])

# --- D. transformer stack --------------------------------------------
box(61, 53.5, 22, 29, NAVY_LIGHT, NAVY)
caption(72, 81.0, "Transformer blocks", NAVY)

# block 1
box(62, 54.8, 20, 2.2, WHITE, NAVY_MID, lw=1.0, r=0.4, z=2)
text(72, 55.9, "block 1", size=10, color=INK)
dots_v(72, 58.2)
# block k (highlighted)
box(62, 59.4, 20, 15.0, ORANGE_LIGHT, ORANGE, lw=1.8, r=0.6, z=2)
text(62.8, 73.3, "block k", size=10, color=ORANGE, weight="bold", ha="left")
XS = 70.5  # residual stream x
# stream through block k
line([(XS, 59.4), (XS, 61.2)], color=NAVY, lw=1.4, z=3)
box(65.5, 61.2, 10, 2.8, WHITE, NAVY, lw=1.1, r=0.4, z=3)
text(70.5, 62.6, "attention", size=10, color=INK, z=7)
line([(XS, 64.0), (XS, 64.85)], color=NAVY, lw=1.4, z=3)
ax.add_patch(Circle((XS, 65.4), 0.55, fc=WHITE, ec=NAVY, lw=1.1, zorder=6))
text(XS, 65.4, "+", size=10, color=NAVY, weight="bold", z=7)
line([(XS, 65.95), (XS, 67.2)], color=NAVY, lw=1.4, z=3)
box(65.5, 67.2, 10, 2.8, WHITE, NAVY, lw=1.1, r=0.4, z=3)
text(70.5, 68.6, "MLP", size=10, color=INK, z=7)
line([(XS, 70.0), (XS, 70.85)], color=NAVY, lw=1.4, z=3)
ax.add_patch(Circle((XS, 71.4), 0.55, fc=WHITE, ec=NAVY, lw=1.1, zorder=6))
text(XS, 71.4, "+", size=10, color=NAVY, weight="bold", z=7)
arrow((XS, 71.95), (XS, 75.3), color=NAVY, lw=1.4, ms=11, z=3)
# residual bypasses
poly_arrow([(XS, 60.3), (78.0, 60.3), (78.0, 65.4), (71.15, 65.4)],
           color=NAVY, lw=1.1, ms=9, z=3)
poly_arrow([(XS, 66.55), (78.0, 66.55), (78.0, 71.4), (71.15, 71.4)],
           color=NAVY, lw=1.1, ms=9, z=3)
text(80.4, 66.0, "residual", size=10, color=NAVY, rotation=90)
# dots + block L
dots_v(72, 76.55)
box(62, 77.4, 20, 2.2, WHITE, NAVY_MID, lw=1.0, r=0.4, z=2)
text(72, 78.5, "block L", size=10, color=INK)

# tap of the residual stream after block k -> hidden states box
ax.scatter([XS], [73.2], s=28, color=ORANGE, zorder=8, lw=0)
poly_arrow([(XS, 73.2), (84.5, 73.2), (84.5, YC), (85.8, YC)], color=ORANGE,
           lw=1.6, ms=13, z=7)

# --- E. hidden states at layer k --------------------------------------
box(86, 59, 13, 18, NAVY_LIGHT, NAVY)
caption(92.5, 75.3, "Hidden states", NAVY)
caption(92.5, 73.6, "at layer k", NAVY)
hs = [shade(NAVY, 5, 0.25, 1.0) for _ in range(6)]
cell_grid(89.6, 62.6, 5, 6, 1.4, 1.35, hs)
text(88.5, 66.65, "T tokens", size=10, color=GREY, rotation=90)
text(93.1, 61.0, "d = 4,096 for Qwen3-8B", size=10, color=GREY)

arrow((99.2, YC), (101.8, YC))

# --- F. mean over tokens ---------------------------------------------
box(102, 62, 13, 12, NAVY_LIGHT, NAVY)
caption(108.5, 72.2, "Mean over", NAVY)
caption(108.5, 70.5, "tokens", NAVY)
text(108.5, 67.2, r"$\bar{h}^{(k)}=\frac{1}{T}\sum_{t=1}^{T} h^{(k)}_t$", size=11.5,
     color=INK)
text(108.5, 63.9, "computed in fp32", size=10, color=GREY)

arrow((115.2, YC), (117.8, YC))

# --- G. one vector per sentence ---------------------------------------
box(118, 57, 12, 22, NAVY_LIGHT, NAVY)
caption(124, 77.2, "One vector", NAVY)
caption(124, 75.5, "per sentence", NAVY)
vec = [[c] for c in shade(NAVY, 10, 0.25, 1.0)]
cell_grid(123.3, 60.2, 1, 10, 2.0, 1.25, vec)
text(121.6, 66.45, "d dims", size=10, color=GREY, rotation=90)

# ---------------------------------------------------------------- row 1 -> row 2
poly_arrow([(124, 57), (124, 47), (15.5, 47), (15.5, 40.2)], color=TEAL, lw=1.6, ms=13)
text(69.5, 48.8, "one vector for every (language, sentence) cell, at every layer",
     size=10, color=TEAL)

# ================================================================ ROW 2
# --- H. the grid -------------------------------------------------------
box(4, 6, 23, 34, TEAL_LIGHT, TEAL)
caption(15.5, 38.3, "Grid of sentence vectors", TEAL)
text(18.25, 35.7, "300 matched sentences", size=10, color=INK)

gx0, gy0, gcw, gch = 10.0, 12.5, 3.3, 4.0
ncol, nrow = 5, 5
# column headers
col_labels = [r"$s_1$", r"$s_2$", r"$s_3$", None, r"$s_{300}$"]
for j, lab in enumerate(col_labels):
    xc = gx0 + (j + 0.5) * gcw
    if lab is None:
        dots_h(xc, 33.7)
    else:
        text(xc, 33.7, lab, size=10, color=GREY)
row_labels = ["eng", "deu", "arb", None, r"$l_{128}$"]
for i, lab in enumerate(row_labels):
    yc = gy0 + (nrow - 1 - i + 0.5) * gch
    if lab is None:
        dots_v(9.3, yc)
    else:
        text(9.6, yc, lab, size=10, color=GREY, ha="right")
text(5.3, gy0 + nrow * gch / 2, "128 languages", size=10, color=INK, rotation=90)

# cells
for i in range(nrow):
    for j in range(ncol):
        x = gx0 + j * gcw
        y = gy0 + (nrow - 1 - i) * gch
        ax.add_patch(Rectangle((x, y), gcw, gch, fc=WHITE, ec=TEAL_MID, lw=0.8, zorder=2))
        if i == 3 or j == 3:
            if i == 3 and j == 3:
                dots_v(x + gcw / 2, y + gch / 2)
            elif i == 3:
                dots_v(x + gcw / 2, y + gch / 2)
            else:
                dots_h(x + gcw / 2, y + gch / 2)
        else:
            cols = [[c] for c in shade(TEAL, 4, 0.3, 1.0)]
            cell_grid(x + gcw / 2 - 0.45, y + 0.6, 1, 4, 0.9, 0.7, cols, z=3)
text(15.5, 10.4, "each cell holds one", size=10, color=GREY)
text(15.5, 8.6, "d-dim sentence vector", size=10, color=GREY)

arrow((27.2, 32.2), (29.8, 32.2), color=TEAL)

# --- I. standardize ---------------------------------------------------
box(30, 26, 18, 12.5, TEAL_LIGHT, TEAL)
caption(39, 36.6, "Standardize", TEAL)
text(39, 34.3, "z-score each of the", size=10)
text(39, 32.5, "d coordinates", size=10)
text(39, 30.7, "over the whole grid", size=10)
text(39, 28.4, "(mean 0, sd 1 each)", size=10, color=GREY)

arrow((39, 25.8), (39, 19.8), color=TEAL)

# --- J. marginal means -----------------------------------------------
box(30, 6, 18, 13.5, TEAL_LIGHT, TEAL)
caption(39, 17.6, "Marginal means", TEAL)
text(39, 15.2, r"language means  $\bar{h}_l$", size=10.5)
text(39, 13.0, r"sentence means  $\bar{h}_c$", size=10.5)
text(39, 10.8, r"grand mean  $\bar{h}$", size=10.5)
text(39, 8.4, "(after standardizing)", size=10, color=GREY)

# means -> SS boxes
arrow((48.2, 13.5), (50.8, 13.5), color=TEAL)
poly_arrow([(48.2, 12.0), (49.5, 12.0), (49.5, 30.5), (50.8, 30.5)], color=TEAL)

# --- K. two sums of squares ------------------------------------------
box(51, 25, 23, 11, TEAL_LIGHT, TEAL)
caption(62.5, 33.9, "Language sum of squares", TEAL)
text(62.5, 30.2, r"$SS_{\mathrm{lang}} = N\,\sum_{l=1}^{L}\ \|\bar{h}_l-\bar{h}\|^{2}$",
     size=11.5)
text(62.5, 26.9, "L = 128 languages", size=10, color=GREY)

box(51, 8, 23, 11, TEAL_LIGHT, TEAL)
caption(62.5, 16.9, "Content sum of squares", TEAL)
text(62.5, 13.2, r"$SS_{\mathrm{con}} = L\,\sum_{c=1}^{N}\ \|\bar{h}_c-\bar{h}\|^{2}$",
     size=11.5)
text(62.5, 9.9, "N = 300 sentences", size=10, color=GREY)

text(62.5, 22.0, "residual term excluded", size=10, color=GREY, style="italic")

# SS -> LFS (merge)
line([(74.2, 30.5), (75.5, 30.5), (75.5, 22.0)], color=TEAL)
line([(74.2, 13.5), (75.5, 13.5), (75.5, 22.0)], color=TEAL)
arrow((75.5, 22.0), (76.8, 22.0), color=TEAL)

# --- L. LFS -----------------------------------------------------------
box(77, 16, 20, 12, ORANGE_LIGHT, ORANGE, lw=1.8)
caption(87, 26.2, "Language-Factor Share", ORANGE)
text(87, 21.6,
     r"$\mathrm{LFS}=\dfrac{SS_{\mathrm{lang}}}{SS_{\mathrm{lang}}+SS_{\mathrm{con}}}$",
     size=12.5)
text(87, 17.8, "one number per layer", size=10, color=GREY)

arrow((97.2, 22.0), (106.8, 22.0), color=ORANGE, lw=1.6)
text(102, 25.5, "repeat at", size=10, color=ORANGE)
text(102, 23.7, "every layer", size=10, color=ORANGE)

# --- M. depth profile --------------------------------------------------
box(107, 6, 23, 34, TEAL_LIGHT, TEAL)
caption(118.5, 38.3, "LFS depth profile", TEAL)

# inset axes in figure coordinates
ix0, iy0, iw, ih = 111.5, 10.5, 17.0, 23.5
ins = fig.add_axes([ix0 / W, iy0 / H, iw / W, ih / H])
layers = np.linspace(0, 32, 300)
KMIN = 11.5
# interior minimum with recovery at both ends (two quadratics, zero slope at the minimum)
prof = np.where(layers < KMIN,
                0.335 + 0.165 * ((KMIN - layers) / KMIN) ** 2,
                0.335 + 0.135 * ((layers - KMIN) / (32 - KMIN)) ** 2)
ins.plot(layers, prof, color=TEAL, lw=2.2, solid_capstyle="round")
ins.scatter([KMIN], [prof.min()], s=42, color=ORANGE, zorder=5, lw=0)
ins.axhline(0.298, color=GREY, ls=(0, (2.5, 2.5)), lw=1.3)
ins.text(31.8, 0.286, "pure-noise value", ha="right", va="top", fontsize=10, color=GREY)
ins.text(31.8, 0.259, "0.298", ha="right", va="top", fontsize=10, color=GREY)
ins.annotate("interior\nminimum", xy=(KMIN, prof.min() + 0.006), xytext=(14.5, 0.50),
             fontsize=10, color=ORANGE, ha="center", va="center",
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1, mutation_scale=10,
                             shrinkB=4))
ins.set_xlim(-0.5, 32.5)
ins.set_ylim(0.225, 0.56)
ins.set_xticks([0, 32])
ins.set_xticklabels(["0", "K"])
ins.set_yticks([])
ins.tick_params(axis="x", labelsize=10, length=3, colors=GREY, labelcolor=GREY)
for sp in ("top", "right"):
    ins.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ins.spines[sp].set_color(GREY)
ins.set_xlabel("layer", fontsize=10, color=INK, labelpad=2)
ins.text(-0.02, 1.01, "LFS", transform=ins.transAxes, ha="left", va="bottom",
         fontsize=10, color=INK)
ins.set_facecolor(WHITE)

# ---------------------------------------------------------------- save
os.makedirs(os.path.dirname(OUT_MAIN), exist_ok=True)
os.makedirs(os.path.dirname(OUT_COPY), exist_ok=True)
fig.savefig(OUT_MAIN, dpi=200, facecolor=WHITE)
shutil.copyfile(OUT_MAIN, OUT_COPY)
print(OUT_MAIN)
print(OUT_COPY)
