"""Shared style for the ICLR figures: one message per chart, few marks, direct labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_names = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"] if f in _names), "DejaVu Sans")
BLUE, BLUE_L, BAND, ORANGE, GREY, GREY_L, INK, INK2 = "#1f5fbf", "#9dbde9", "#d6e4f7", "#e4572e", "#8c8c8c", "#d9d9d9", "#1f1f1f", "#555555"
def setup(size=8.5):
    plt.rcParams.update({"font.family": FONT, "font.size": size, "axes.labelsize": size, "axes.titlesize": size,
                         "axes.titleweight": "medium", "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
                         "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "xtick.color": "#444444", "ytick.color": "#444444",
                         "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "axes.titlelocation": "left", "axes.titlepad": 6, "legend.frameon": False, "legend.fontsize": size - 1,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic"})
def clean(ax, grid=None):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if grid: ax.grid(axis=grid, color="#ececec", lw=0.7)
def save(fig, stem):
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("wrote", stem + ".{pdf,png}")
