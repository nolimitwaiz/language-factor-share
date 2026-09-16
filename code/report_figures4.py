#!/usr/bin/env python3
"""Figures 12 and 13 of the technical report.

These two were originally generated from inline code during report
assembly and were the only report outputs not regenerating from tracked
code (audit finding A7). This script reproduces them from the committed
result files.

  fig15_budget_composition.png  ->  report Figure 12
  fig16_tail_curves.png         ->  report Figure 13

Both read from results/ only; no model is loaded and nothing is extracted.

Usage: python code/report_figures4.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'paper', 'figs')

# Campaign model order, small to large within family. Fixed so the figure
# is stable across regenerations.
ORDER = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base",
         "Qwen3-8B-Base", "OLMo-2-0425-1B", "OLMo-2-1124-7B",
         "Mistral-7B-v0.3", "bloom-1b7", "bloom-7b1", "EuroLLM-1.7B",
         "SmolLM2-1.7B", "Falcon3-7B-Base", "salamandra-2b",
         "salamandra-7b"]

# Tail-curve highlights: one model per recipe class, chosen to span the
# range of tail shapes rather than to flatter any model.
HIGHLIGHT = {"Qwen3-8B-Base": "#0891A8", "OLMo-2-1124-7B": "#7059C6",
             "bloom-1b7": "#B4593A", "SmolLM2-1.7B": "#5A6472"}

QS = [20, 10, 5, 1]


def budget_composition(out=os.path.join(FIGS, 'fig15_budget_composition.png')):
    """Report Figure 12. Uses the fixed-rank (r=64) configuration for every
    model so that shares are comparable across the grid; the selected-rank
    primaries are reported in the table instead."""
    rows = []
    for tag in ORDER:
        p = os.path.join(ROOT, 'results', 'budget', f'{tag}_fixed_raw',
                         'budget.json')
        d = json.load(open(p))
        pl = d['per_language']
        m = {r: float(np.mean([v[f'share_{r}']['mean'] for v in pl.values()]))
             for r in ('M1', 'M2', 'M3', 'M4', 'M5')}
        # Residual is what no systematic transformation explains. Negative
        # rung shares are held-out overfit flags and are floored at zero for
        # display only; the table reports them signed.
        m['resid'] = max(0.0, 1.0 - sum(max(m[k], 0)
                                        for k in ('M1', 'M2', 'M3', 'M4', 'M5')))
        rows.append(m)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    y = np.arange(len(ORDER))[::-1]
    left = np.zeros(len(ORDER))
    specs = [("M1", "Additive offset", "#B4593A"),
             ("M2", "Isotropic scale", "#DCA284"),
             ("M4", "General linear", "#7059C6"),
             ("M3", "Rotation", "#B42318")]
    for key, lab, c in specs:
        vals = np.array([max(r[key], 0) for r in rows])
        ax.barh(y, vals, left=left, height=0.62, color=c, label=lab,
                edgecolor="white", linewidth=0.8)
        left += vals
    ax.barh(y, np.array([r['resid'] for r in rows]), left=left, height=0.62,
            color="#C9CDC6", hatch="///", edgecolor="white", linewidth=0.8,
            label="Unattributed residual")
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("-Base", "") for t in ORDER], fontsize=9)
    ax.set_xlabel("Share of held-out cross-language misalignment "
                  "(common rank r = 64)")
    ax.set_xlim(0, 1.12)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Misalignment decomposition at the dip layer, 14 models "
                 "(n = 1500 sentences/language)", fontsize=11)
    for yi, r in zip(y, rows):
        ax.annotate(f'rotation {100 * r["M3"]:.1f}%', xy=(1.115, yi),
                    fontsize=7, color="#B42318", ha="right", va="center")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[fig] {out}')


def tail_curves(out=os.path.join(FIGS, 'fig16_tail_curves.png')):
    """Report Figure 13. Salamandra-2B is omitted for axis legibility; its
    curve sits far below the others and its values are in the JSON."""
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(QS))
    for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'tailcurves',
                                           '*.json'))):
        if f.endswith('summary.json'):
            continue
        d = json.load(open(f))
        tag = d['model']
        if tag == 'salamandra-2b':
            continue
        pl = d['per_language']
        curve = [float(np.mean([v[f'aar{q}'] for v in pl.values()]))
                 for q in QS]
        if tag in HIGHLIGHT:
            ax.plot(x, curve, color=HIGHLIGHT[tag], lw=2.2, marker="o", ms=5,
                    label=tag.replace("-Base", ""), zorder=3)
        else:
            ax.plot(x, curve, color="#B9BEB8", lw=1.0, zorder=1)
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"AaR@{q}" for q in QS])
    ax.set_xlabel("Tail percentile (worst q percent of sentences per language)")
    ax.set_ylabel("Mean retrieval margin of the tail")
    ax.set_title("Alignment-at-Risk tail curves, mean across 128 languages "
                 "(n = 1500)", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[fig] {out}')


if __name__ == '__main__':
    budget_composition()
    tail_curves()
