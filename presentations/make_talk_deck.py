#!/usr/bin/env python
"""Build the 14-slide LFS talk deck for the Koehn meeting (7 September 2026).

Run with:  ~/.hiero_venv/bin/python make_talk_deck.py

CPU-only, no data or model inference. Every number on a slide traces to
paper/iclr2027/main.tex or docs/KOEHN_MEETING_BRIEF_2026-09-05.md.
Speaker notes carry the detail and the sentences to say.

After saving, the script reopens each deck with python-pptx, counts visible
words per slide (text frames, table cells, grouped shapes), asserts every
slide has at most 60 visible words and every referenced image exists, checks
notes lengths (150 to 250 words), scans for banned wording, and prints the
per-slide counts.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path("/Users/waizkhan/Desktop/multilingual-metrics")
FIGS = ROOT / "paper" / "figs"
ICLR_FIGS = ROOT / "paper" / "iclr2027" / "figs"

IMG = {
    "pipeline": FIGS / "fig22_pipeline_diagram.png",
    "profiles": ICLR_FIGS / "fig1_lfs_profiles_17.png",
    "scatter": ICLR_FIGS / "fig_r1_model_level.png",
    "ratio": FIGS / "fig17_ratio_hides_scale.png",
    "budget": FIGS / "fig15_budget_composition.png",
    "tail": FIGS / "fig16_tail_curves.png",
}

OUT_PATHS = [
    ROOT / "paper" / "slides_pptx" / "Waiz_Khan_LFS_Talk_2026-09-09.pptx",
    ROOT / "deliverables" / "2026-09-09" / "Waiz_Khan_LFS_Talk_2026-09-09.pptx",
    ROOT / "deliverables" / "2026-09-10" / "Waiz_Khan_LFS_Talk_2026-09-10.pptx",
]

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.6
CONTENT_W = SLIDE_W - 2 * MARGIN
TITLE_TOP, TITLE_H = 0.35, 0.8
RULE_TOP = 1.17
CONTENT_TOP = 1.45
CONTENT_BOTTOM = 6.85

FONT = "Calibri"
NAVY = RGBColor(0x1B, 0x2A, 0x4A)
ACCENT = RGBColor(0xD9, 0x6B, 0x27)
TEXT = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
LIGHT = RGBColor(0xF3, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE_GRAY = RGBColor(0xC9, 0xCD, 0xD6)

TITLE_PT = 28
BODY_PT = 20
MIN_BODY_PT = 18
MAX_WORDS = 60
NOTES_MIN, NOTES_MAX = 150, 250

BANNED = [
    "fingerprint", "toy", "baseline", "beats", "u-shaped", "u-shape",
    "language-agnostic", "first of its kind", "—",
]

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def set_run(run, size=BODY_PT, bold=False, italic=False, color=TEXT, sub=False, sup=False):
    f = run.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    if sub:
        f._element.set("baseline", "-25000")
    if sup:
        f._element.set("baseline", "30000")


def add_text(slide, left, top, width, height, paras, size=BODY_PT, color=TEXT,
             bold=False, italic=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             space_after=6, line_spacing=1.08, wrap=True):
    """paras: list of paragraphs. Each paragraph is a str, or a dict with keys
    text | runs, size, bold, italic, color, align, space_after.
    runs: list of (text, {size,bold,italic,color,sub,sup}) tuples."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for p in paras:
        if isinstance(p, str):
            p = {"text": p}
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = p.get("align", align)
        para.space_after = Pt(p.get("space_after", space_after))
        para.line_spacing = line_spacing
        psize = p.get("size", size)
        pbold = p.get("bold", bold)
        pital = p.get("italic", italic)
        pcolor = p.get("color", color)
        if "runs" in p:
            for text, opts in p["runs"]:
                r = para.add_run()
                r.text = text
                set_run(r, size=opts.get("size", psize), bold=opts.get("bold", pbold),
                        italic=opts.get("italic", pital), color=opts.get("color", pcolor),
                        sub=opts.get("sub", False), sup=opts.get("sup", False))
        else:
            r = para.add_run()
            r.text = p["text"]
            set_run(r, size=psize, bold=pbold, italic=pital, color=pcolor)
    return box


def add_rect(slide, left, top, width, height, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    shp = slide.shapes.add_shape(shape, Inches(left), Inches(top), Inches(width), Inches(height))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    # no text in decorative shapes
    shp.text_frame.text = ""
    return shp


def add_title(slide, text):
    add_text(slide, MARGIN, TITLE_TOP, CONTENT_W, TITLE_H, [text], size=TITLE_PT,
             color=NAVY, bold=True, anchor=MSO_ANCHOR.BOTTOM, space_after=0)
    add_rect(slide, MARGIN, RULE_TOP, 1.2, 0.06, ACCENT)


def add_slide_number(slide, n):
    add_text(slide, SLIDE_W - MARGIN - 0.7, 6.98, 0.7, 0.35, [str(n)], size=12, color=GRAY,
             align=PP_ALIGN.RIGHT, space_after=0)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def fit_picture(slide, path, left, top, max_w, max_h, center=True):
    with Image.open(path) as im:
        w_px, h_px = im.size
    aspect = w_px / h_px
    w, h = max_w, max_w / aspect
    if h > max_h:
        h, w = max_h, max_h * aspect
    if center:
        left = left + (max_w - w) / 2
        top = top + (max_h - h) / 2
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(w), Inches(h))


def sub_runs(base, subscript, **opts):
    """Return runs for e.g. SS with subscript lang."""
    return [(base, dict(opts)), (subscript, dict(opts, sub=True))]


def new_slide(prs, title, n):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE
    if title:
        add_title(slide, title)
    add_slide_number(slide, n)
    return slide


# --------------------------------------------------------------------------
# Slide builders
# --------------------------------------------------------------------------


def slide_01_title(prs):
    s = new_slide(prs, None, 1)
    add_rect(s, MARGIN, 2.05, 1.6, 0.08, ACCENT)
    add_text(s, MARGIN, 2.25, CONTENT_W, 1.1, ["Language-Factor Share"], size=44, color=NAVY,
             bold=True, space_after=0)
    add_text(s, MARGIN, 3.3, CONTENT_W, 0.7, ["Measuring multilingual representation structure"],
             size=28, color=NAVY, space_after=0)
    add_text(s, MARGIN, 4.25, CONTENT_W, 0.6,
             ["NSF Aim 1 progress and an ICLR 2027 submission plan"], size=22, color=TEXT,
             space_after=0)
    add_text(s, MARGIN, 5.3, CONTENT_W, 0.5,
             ["Waiz Khan, Johns Hopkins University, 7 September 2026"], size=18, color=GRAY,
             space_after=0)
    add_notes(s, """
Thank you for the time. I will take about twenty minutes, and I would like to leave with three decisions: the framing and title, the author list and who registers as a reviewer, and a yes to submit. Everything else is detail you can read in the draft.

The paper is the Aim 1 deliverable. It proposes one measurement, the Language-Factor Share, or LFS, establishes it on 33 open models, validates it against behavior after covariate adjustment, and maps where it fails. The title on this slide is the working title; I want your view on it at the end.

Here is the one sentence. LFS measures, at every layer, how much of the systematic variation in a model's hidden states follows the language of a sentence rather than its meaning, on a grid of matched translations. The paper establishes the measurement, validates it against behavior, and maps where it fails.

Two framing notes before I start. First, the finance framing you found obscuring in July is gone from the paper; LFS is described as what it is, a two-way variance decomposition. Second, the paper ranks no metric against another. MEXA is a published reference measure computed alongside our measures, and nothing is ranked against it. This is not a leaderboard paper.

The draft PDF, the revised technical report, and the two-page explainer are attached to the calendar invitation. I will open the draft to Figure 1 on page 2 as we go.
""")


def slide_02_question(prs):
    s = new_slide(prs, "The question in one sentence", 2)
    add_text(s, MARGIN, CONTENT_TOP, CONTENT_W, 1.15,
             ["How much systematic variation in hidden states follows language rather than "
              "meaning, at each layer?"], size=24, color=NAVY, space_after=0)

    # Worked table: one coordinate, two languages, three sentences.
    rows, cols = 5, 4
    tbl_left, tbl_top, tbl_w, tbl_h = MARGIN, 2.75, 6.4, 3.0
    gt = s.shapes.add_table(rows, cols, Inches(tbl_left), Inches(tbl_top), Inches(tbl_w), Inches(tbl_h))
    table = gt.table
    # Neutral styling: remove banded look by setting fills explicitly.
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    data = [
        ["", "A", "B", "Mean"],
        ["Sentence 1", "2", "5", "3.5"],
        ["Sentence 2", "6", "9", "7.5"],
        ["Sentence 3", "10", "13", "11.5"],
        ["Mean", "6", "9", "7.5"],
    ]
    col_w = [2.0, 1.3, 1.3, 1.8]
    for j, w in enumerate(col_w):
        table.columns[j].width = Inches(w)
    for i in range(rows):
        table.rows[i].height = Inches(tbl_h / rows)
        for j in range(cols):
            cell = table.cell(i, j)
            cell.fill.solid()
            header = (i == 0) or (j == 0) or (i == rows - 1) or (j == cols - 1)
            cell.fill.fore_color.rgb = LIGHT if header else WHITE
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = data[i][j]
            is_mean = (i == rows - 1) or (j == cols - 1)
            set_run(r, size=MIN_BODY_PT, bold=header,
                    color=(ACCENT if (is_mean and data[i][j] not in ("Mean", "")) else (NAVY if header else TEXT)))

    # Formulas at right.
    fx = tbl_left + tbl_w + 0.5
    fw = SLIDE_W - MARGIN - fx
    add_text(s, fx, 2.85, fw, 2.6, [
        {"runs": sub_runs("SS", "lang") + [(" = 13.5", {})], "size": 22},
        {"runs": sub_runs("SS", "con") + [(" = 64", {})], "size": 22},
        {"runs": [("LFS = 13.5/77.5 = ", {}), ("0.17", {"bold": True, "color": ACCENT})], "size": 22},
    ], color=NAVY, space_after=14)
    add_text(s, fx, 5.55, fw, 0.5, ["An illustration, not model data."], size=MIN_BODY_PT,
             color=GRAY, italic=True, space_after=0)
    add_notes(s, """
The question in one sentence: at each layer, how much of the systematic variation in hidden states follows which language a sentence is in, rather than which sentence it is?

The table is the whiteboard version: one coordinate, two languages, three sentences. It is an illustration, not model data. Language A has the values 2, 6, 10 and language B has 5, 9, 13. The language means are 6 and 9. The sentence means are 3.5, 7.5, and 11.5. The grand mean is 7.5.

The language sum of squares is the number of sentences, three, times the squared distance of each language mean from the grand mean: 3 times 2.25 plus 2.25, which is 13.5. The concept sum of squares is the number of languages, two, times the squared distance of each sentence mean from the grand mean: 2 times 16 plus 0 plus 16, which is 64. LFS is 13.5 over 77.5, about 0.17. Most of the systematic variation here follows the sentence, not the language.

In the real measurement every entry is a 4,096-dimensional vector, so squared differences become squared norms, and there are 128 languages and 300 sentences instead of two and three. Nothing else changes. If someone says this is analysis of variance: yes, a balanced two-way layout with one observation per cell. The contribution is the operationalization on translation grids, the null, 33 models, confounder-adjusted validation, and the measured failure modes.
""")


def slide_03_pipeline(prs):
    s = new_slide(prs, "What happens inside the model", 3)
    img_top, img_h = 1.35, 5.45
    pic = fit_picture(s, IMG["pipeline"], MARGIN, img_top, 8.3, img_h, center=False)
    cap_left = MARGIN + 8.3 + 0.4
    cap_w = SLIDE_W - MARGIN - cap_left
    add_text(s, cap_left, 1.9, cap_w, 4.6, [
        {"runs": [("Tap ", {}), ("the residual stream after block k; mean-pool tokens in fp32.", {})]},
        "One 4,096-d vector per (language, sentence) cell, at every layer.",
        "Standardize jointly; split systematic variation into language and sentence means.",
    ], size=BODY_PT, color=TEXT, space_after=16)
    add_notes(s, """
This is the whole pipeline in one picture. The top row is inside the model. Take one sentence, its German translation, and its Arabic translation. Each string goes through the tokenizer and the token embeddings, then through the stack of Transformer blocks. At block k, highlighted in orange, we tap the residual stream after the block. That gives a T by 4,096 matrix of hidden states for a sentence of T tokens. We mean-pool over the non-padding tokens in fp32, and we get one 4,096-dimensional vector per sentence, per layer.

The connector carries the point that matters: one vector for every language and sentence cell, at every layer.

The bottom row is the grid and the decomposition. The vectors fill a 128-language by 300-sentence table. We standardize each coordinate jointly over the whole grid, mean zero and standard deviation one, so a few high-variance coordinates do not dominate the sums. Standardization is joint, never per language; per-language standardization would remove the quantity being measured. Then we take language means, sentence means, and the grand mean, form the two sums of squares, and read the share. The residual term is excluded from the ratio and reported beside it.

We repeat at every layer and plot the profile. The sketch at the right is what we see: an interior minimum with both ends higher, and a dotted line at the pure-noise value 0.298. Extraction is the only GPU work; everything after the tap is CPU arithmetic on saved dumps.
""")


def slide_04_definition(prs):
    s = new_slide(prs, "Definition and the pure-noise value", 4)
    add_text(s, MARGIN, CONTENT_TOP, CONTENT_W, 0.9, [
        "128 languages × 300 matched NTREX sentences; tokens mean-pooled; coordinates "
        "standardized jointly over the grid."
    ], size=BODY_PT, color=TEXT, space_after=0)

    card_top, card_h = 2.55, 2.55
    add_rect(s, MARGIN, card_top, CONTENT_W, card_h, LIGHT)
    add_text(s, MARGIN + 0.4, card_top + 0.2, CONTENT_W - 0.8, card_h - 0.4, [
        {"runs": sub_runs("SS", "lang") + [(" = N Σ", {}), ("ℓ", {"sub": True}),
                                          (" ‖z̄", {}), ("ℓ", {"sub": True}),
                                          (" − z̄‖²", {})], "size": 24},
        {"runs": sub_runs("SS", "con") + [(" = L Σ", {}), ("c", {"sub": True}),
                                         (" ‖z̄", {}), ("c", {"sub": True}),
                                         (" − z̄‖²", {})], "size": 24},
        {"runs": [("LFS = ", {"bold": True})] + sub_runs("SS", "lang", bold=True)
                 + [("/(", {"bold": True})] + sub_runs("SS", "lang", bold=True)
                 + [(" + ", {"bold": True})] + sub_runs("SS", "con", bold=True)
                 + [(")", {"bold": True})], "size": 26, "color": ACCENT},
    ], color=NAVY, space_after=12, anchor=MSO_ANCHOR.MIDDLE)

    add_text(s, MARGIN, 5.4, CONTENT_W, 1.2, [
        {"runs": [("Pure noise gives (L−1)/(L+N−2) = ", {}),
                  ("0.298", {"bold": True, "color": ACCENT}),
                  (" on this grid. Every measured value sits far above it.", {})]},
    ], size=BODY_PT, color=TEXT, space_after=0)
    add_notes(s, """
The definition, precisely. Take N sentences translated into L languages. At one layer, mean-pool the token hidden states in fp32 and get one vector per cell of a balanced L by N grid. Standardize each coordinate jointly over all L times N cells.

SS_lang is N times the sum over languages of the squared norm of the centered language mean. SS_con is L times the sum over sentences of the squared norm of the centered sentence mean. On a balanced grid these two terms and the residual are mutually orthogonal, so total variation equals SS_lang plus SS_con plus the residual, exactly. LFS is SS_lang over SS_lang plus SS_con.

The residual is deliberately excluded. LFS asks how the two systematic main effects divide, so that a model's idiosyncratic noise level does not masquerade as shared structure. The residual, and the raw magnitudes of all three terms, are reported beside the ratio as part of the measurement.

The pure-noise value is the calibration. Under an additive model with isotropic noise and no language or concept structure, the estimator returns the degrees-of-freedom ratio, L minus 1 over L plus N minus 2, which is 0.298 at 128 languages and 300 sentences. Synthetic noise across nine configurations and pseudo-language splits of real representations reproduce it to three decimals. Every measured value sits far above it, and in fact above 0.5, so even at the dip layer the measure is relative. Absolute values are comparable only at fixed grid, pooling, and standardization.
""")


def slide_05_profiles(prs):
    s = new_slide(prs, "Every profile dips and recovers", 5)
    fit_picture(s, IMG["profiles"], MARGIN, 1.4, CONTENT_W, 4.55)
    add_text(s, MARGIN, 6.05, CONTENT_W, 0.8, [
        "Same 300 sentences, 128 languages. High at the input, lowest mid-network, high again at "
        "the output: 17 of 17 here, 19 of 19 in the expansion set. Dotted line: pure-noise "
        "value 0.298."
    ], size=MIN_BODY_PT, color=TEXT, space_after=0)
    add_notes(s, """
This is Figure 1 of the draft. Same 300 sentences in 128 languages, LFS by relative depth for the 17 primary-set models, four highlighted and the rest in gray. High at the input, lowest mid-network, high again at the output, in all 17 models here and in 19 more measured with a restricted-maximum-likelihood variant in the expansion set. On the 14 models measured by both estimators the dip-depth rankings agree at Spearman 1.000.

Every value lies far above the grid's pure-noise value of 0.298, the dotted line. The minimum lies at relative depth 0.06 to 0.66, most between 0.35 and 0.6. Three models, Mistral-7B and the continued-pretraining models TowerBase-7B and occiglot-7b-eu5, place a sharp minimum at layer 2 followed by a broad plateau, which no summary of the form "middle layers are less language dependent" predicts.

If you say this confirms what we already knew, the answer is yes for the shape. What is new is the number and what it does: a comparable depth with a nine-fold spread, recipe over scale, three models with a minimum at layer 2, script and meaning leaving on different schedules, and a validated relationship to cross-lingual context use.

The right-hand panel is dip depth, layer-0 LFS minus the minimum, for all 17 models; that is the next slide. One caution in wording: the paper says interior minimum with endpoint recovery and does not test for a formally smooth shape.
""")


def slide_06_depth(prs):
    s = new_slide(prs, "Dip depth: recipe over scale", 6)
    add_text(s, MARGIN, CONTENT_TOP, CONTENT_W, 0.9, [
        "Dip depth = first-layer LFS minus the minimum. Nine-fold spread across 17 models, "
        "0.039 to 0.336."
    ], size=BODY_PT, color=TEXT, space_after=0)

    bars = [("Qwen3-8B", 0.336), ("Llama-3.1-8B", 0.335), ("Qwen3-1.7B", 0.213),
            ("Mistral-7B", 0.212), ("BLOOM-1.7B", 0.039)]
    top0, row_h, bar_h = 2.5, 0.72, 0.42
    label_w, bar_left, bar_max_w = 2.1, MARGIN + 2.2, 4.0
    for i, (name, val) in enumerate(bars):
        y = top0 + i * row_h
        add_text(s, MARGIN, y, label_w, bar_h, [name], size=MIN_BODY_PT, color=NAVY,
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        w = max(0.06, bar_max_w * val / 0.336)
        add_rect(s, bar_left, y, w, bar_h, ACCENT if name != "BLOOM-1.7B" else NAVY)
        add_text(s, bar_left + w + 0.1, y, 1.0, bar_h, [f"{val:.3f}"], size=MIN_BODY_PT,
                 color=TEXT, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    # thin axis line
    add_rect(s, bar_left, top0 - 0.08, bar_max_w + 0.2, 0.02, RULE_GRAY)

    tx = MARGIN + 7.9
    add_text(s, tx, 2.5, SLIDE_W - MARGIN - tx, 3.6, [
        "Qwen3-1.7B dips deeper than every other lab's 7B model except Llama-3.1-8B.",
        "BLOOM, deliberately balanced over 46 languages, is shallowest at both sizes.",
    ], size=BODY_PT, color=TEXT, space_after=18)
    add_notes(s, """
Dip depth is the first-layer LFS minus the minimum. It ranges from 0.039 for BLOOM-1.7B to 0.336 for Qwen3-8B, a nine-fold spread across the 17 primary models, and it tracks the training recipe more than the size.

Within the Qwen3 series the dip deepens monotonically with size: 0.206, 0.213, 0.314, 0.336 from 0.6B to 8B. But the spread across families at fixed size dwarfs that slope. Qwen3-1.7B at 0.213 dips deeper than every 7B model from another lab except Llama-3.1-8B at 0.335. Mistral-7B sits at 0.212 with its minimum at layer 2. BLOOM, the only deliberately balanced 46-language model in the grid, has the shallowest profile and does not deepen from 1.7B to 7B, 0.039 to 0.043. Salamandra dips more at 2B, 0.254, than at 7B, 0.180, while its retrieval alignment improves, so dip depth is not a monotone scaling proxy.

The claim is descriptive. Two unseen families were run under frozen predictions: SmolLM2-1.7B dipped to 0.068 against a prediction of at most 0.15, and Falcon3-7B to 0.140 against a prediction of intermediate.

On pooling, which you asked about in August: absolute depth moves by 0.13 to 0.20 across mean, final-token, unit-norm, and length-matched pooling, the cross-model ordering changes by at most one adjacent transposition, and the within-family result holds for two Qwen3 sizes that share a tokenizer.
""")


def slide_07_stable(prs):
    s = new_slide(prs, "It replicates and it is stable", 7)
    cards = [
        ("Second corpus",
         "FLORES-200, 116 languages: 17 of 17 recover, dip depths agree at Spearman 0.917."),
        ("Sentence bootstrap",
         "2,000 draws: 95% half-widths 0.004 to 0.020 for 12 of 14 models."),
        ("Instruction tuning",
         "Six base and instruct pairs: dip depth moves at most 0.008."),
    ]
    gap = 0.35
    card_w = (CONTENT_W - 2 * gap) / 3
    card_top, card_h = 1.9, 3.9
    for i, (head, body) in enumerate(cards):
        x = MARGIN + i * (card_w + gap)
        add_rect(s, x, card_top, card_w, card_h, LIGHT)
        add_rect(s, x, card_top, card_w, 0.08, ACCENT)
        add_text(s, x + 0.3, card_top + 0.35, card_w - 0.6, 0.7, [head], size=22, color=NAVY,
                 bold=True, space_after=0)
        add_text(s, x + 0.3, card_top + 1.15, card_w - 0.6, card_h - 1.4, [body], size=BODY_PT,
                 color=TEXT, space_after=0)
    add_notes(s, """
Three reasons to trust the profile as a measurement.

It replicates on a second corpus. We re-measured all 17 models on FLORES-200 devtest, 116 languages, English plus 115 of the 127 non-English languages, 300 sentences, under five frozen predictions. Seventeen of seventeen profiles recover at both endpoints, dip depths agree at Spearman 0.917 with a median absolute difference of 0.011, the minimum layer agrees within two layers for 14 of 17, and per-layer profiles correlate at 0.94 to 1.00.

It is stable under resampling. Sentence-bootstrap intervals on dip depth, 2,000 draws of 300 sentences from the 1,500-sentence dumps with identical index sets for every model, have 95 percent half-widths of 0.004 to 0.020 for 12 of the 14 models. The two wider ones, 0.045 and 0.059, are Mistral-7B and Salamandra-2B, whose minimum sits at layer 2 or 6 where the profile is steep. The cross-model ordering is preserved in 2,000 of 2,000 replicates.

It is a property of pretraining. For six base and instruct pairs, Qwen3 0.6B to 8B, Qwen2.5-7B, and Llama-3.1-8B, dip depth changes by at most 0.008, the minimum layer agrees within two layers for five of six, and the final-layer share differs by at most 0.003. Instruction tuning does not move the profile.

The wording to use is depth profile, measured on a fixed grid, as a structural diagnostic.
""")


def slide_08_budget(prs):
    s = new_slide(prs, "What the remaining differences are", 8)
    img_w = 7.5
    fit_picture(s, IMG["budget"], MARGIN, 1.5, img_w, 4.9)
    tx = MARGIN + img_w + 0.4
    add_text(s, tx, 1.7, SLIDE_W - MARGIN - tx, 4.8, [
        {"runs": [("At the dip layer, additive offset plus isotropic scale explain ", {}),
                  ("63 to 97 percent", {"bold": True, "color": ACCENT}),
                  (" of held-out cross-language misalignment. Rotation: under 1 percent.", {})]},
        {"runs": [("The shared factor is multilingual, not English: it predicts each language "
                   "better than English does for ", {}),
                  ("113 to 124 of 127", {"bold": True, "color": ACCENT}),
                  (" languages.", {})]},
    ], size=BODY_PT, color=TEXT, space_after=20)
    add_notes(s, """
Once most of the variation sits on meaning at the dip layer, what differences between languages remain? Two results.

First, offsets and scale explain most measured misalignment. At each model's dip layer we fit a nested sequence of maps from each language into a generalized-Procrustes consensus: offset, isotropic scale, rotation, general linear, and a nonlinear kernel, each map credited with its held-out reduction in reconstruction error on 20 cross-fitted splits against per-language permutation nulls. Offset plus scale account for 63 to 97 percent of held-out cross-language misalignment across the 14-model study. Rotation accounts for minus 0.2 to plus 0.7 percent, and the instrument recovers injected rotations of 0.1 to 0.8 radians, so the small rotation share means additional orthogonal alignment adds little under this protocol. The general linear map adds 0.4 to 8 percent. A document-purged re-audit on Qwen3-1.7B leaves offset plus scale at about 64 percent, so the legacy nonlinear percentages are upper bounds.

Second, a multilingual average predicts each language better than an English donor. For each target language we compare the cross-validated R-squared of predicting its dip-layer representations from English against predicting them from the mean of all other languages. The multilingual latent factor predicts better than English does for 113 to 124 of 127 languages depending on the model, 124 for most and BLOOM lowest, by a median of plus 0.06 to plus 0.16 R-squared. English is the best single donor for 9 of 10 high-resource targets but 1 of 8 low-resource ones.
""")


def slide_09_content(prs):
    s = new_slide(prs, "Behavior: cross-lingual content transfer", 9)
    img_w = 6.6
    fit_picture(s, IMG["scatter"], MARGIN, 1.45, img_w, 5.2)
    tx = MARGIN + img_w + 0.4
    add_text(s, tx, 1.7, SLIDE_W - MARGIN - tx, 4.9, [
        "Does context in another language help a model predict English text? Content benefit: "
        "matched minus wrong-document context, normalized by English.",
        {"runs": [("1 − LFS at the dip predicts R", {}), ("content", {"sub": True}),
                  (" at Spearman ", {}), ("0.508", {"bold": True, "color": ACCENT}),
                  (", model-bootstrap CI [0.19, 0.71]; 19 frozen models, 13 families; positive "
                   "under every leave-one-family-out deletion.", {})]},
    ], size=BODY_PT, color=TEXT, space_after=20)
    add_notes(s, """
The behavior most directly tied to a shared representation is using information given in one language to predict text in another. From NTREX we take an English target sentence and its preceding same-document sentence, and measure the target's teacher-forced negative log-likelihood with no context, with the preceding sentence in each of 41 context languages, and with a length-matched wrong-document context in the same language. The content benefit is the mismatched NLL minus the matched NLL, normalized by the English benefit. The mismatched control matters: wrong-document foreign text hurts prediction in most languages, so raw context benefits understate content transfer.

A four-model pilot gave 0.84 but cannot support a model-level claim. So we froze a protocol on 19 new models from 13 families and ran it: 157,700 scored sequences.

The sentence to say: across 19 models and 13 families, models with a smaller language share at the dip layer make more use of foreign-language context when predicting English text, after controlling for data size, family, script, and fertility. Confounder-adjusted Spearman 0.508, model-bootstrap interval 0.19 to 0.71, positive under every leave-one-family-out deletion, 0.770 at the model level. Each point in the figure is one model, placed by its sentence-component measure at the dip and its median normalized content benefit over 34 languages.

This is a predictive association, not causation. MEXA has a larger point estimate on the same test, 0.591; the two answer different questions and are not ranked.
""")


def slide_10_exam(prs):
    s = new_slide(prs, "Where it fails: pooled benchmark accuracy", 10)
    left_w = 5.2
    add_text(s, MARGIN, 1.7, left_w, 1.4, ["0.041"], size=66, color=ACCENT, bold=True,
             space_after=0)
    add_text(s, MARGIN, 3.15, left_w, 0.6, ["95% CI [−0.05, 0.12]"], size=22, color=NAVY,
             space_after=0)
    add_text(s, MARGIN, 3.9, left_w, 1.6, [
        "Adjusted Spearman, 1 − LFS-VC vs pooled Belebele and INCLUDE accuracy; 33 models."
    ], size=MIN_BODY_PT, color=GRAY, space_after=0)

    tx = MARGIN + left_w + 0.6
    tw = SLIDE_W - MARGIN - tx
    add_rect(s, tx - 0.3, 1.6, 0.06, 4.6, RULE_GRAY)
    add_text(s, tx, 1.7, tw, 4.6, [
        {"text": "The reliability ceiling", "size": 22, "bold": True, "color": NAVY, "space_after": 10},
        "Two independent benchmarks agree on only 0.315 of their residual variation, so pooled "
        "scores are a noisy criterion. Averaging the two recovers a positive interval.",
        {"text": "A scope condition, not a defect.", "bold": True, "color": NAVY},
    ], size=BODY_PT, color=TEXT, space_after=16)
    add_notes(s, """
Now the two boundaries, before they are raised. The first is pooled benchmark accuracy.

Raw correlations between intrinsic alignment and Belebele accuracy are moderate, but the correlation decreases after adjustment for exposure-related covariates. We fit a confounder adjustment model on the logit scale: model fixed effects, log training tokens from CulturaX, macro-family, script, and tokenizer fertility, R-squared 0.86 with a positive and significant token coefficient. Validity is the Spearman correlation between residualized metric and residualized outcome, with cluster bootstrap intervals by language, by model, and crossed.

On the 33-model panel (1,740 rows, 70 languages), the sentence-component share, one minus LFS-VC, has covariate-adjusted Spearman 0.041 with interval minus 0.05 to 0.12. There is no clear adjusted association with pooled benchmark accuracy. Neither does anything else strongly: MEXA is 0.238 and the tail measure 0.105 on the same panel.

The measured reason is the target. Two independent benchmarks of the same 420 cells agree on only 0.315 of their adjusted residual rank variation, which bounds how closely any predictor can track one benchmark's residual. Averaging the two benchmarks recovers a positive interval, plus 0.10 with interval 0.003 to 0.18. On 30 natively authored INCLUDE languages the picture is the same, and the loss traces to that panel's narrow resource range, not to translationese.

Own this: it is a scope condition, not a defect to be tuned away. The July benchmark-capable-subset number of about 0.59 has no stored artifact and is withdrawn; it re-derives at about 0.49.
""")


def slide_11_collapse(prs):
    s = new_slide(prs, "The collapse blind spot", 11)
    fit_picture(s, IMG["ratio"], MARGIN, 1.35, CONTENT_W, 4.35)
    add_text(s, MARGIN, 5.75, CONTENT_W, 1.1, [
        {"runs": [("The proposal's word-alignment MSE loss shrank representation norm 451 to 42 and "
                   "raw concept variance ", {}),
                  ("13,791 to 699", {"bold": True, "color": ACCENT}),
                  (". LFS moved 0.445 to 0.493; retrieval measures rose for the collapsed model, which posted among the highest scores "
                   "first. Only the raw concept-variance measure (CVP) caught it.", {})]},
    ], size=MIN_BODY_PT, color=TEXT, space_after=0)
    add_notes(s, """
The second boundary is one every scale-invariant measure shares. Any statistic unchanged when all representations are multiplied by the same nonzero constant is unchanged when all representations shrink together. Injecting calibrated shrinkage toward the global centroid into saved representations leaves LFS unchanged to one part in a million, and leaves MEXA and the tail measure flat even at 90 percent collapse, because retrieval is order-based. Only a raw concept-variance preservation check, CVP, concept variance relative to a frozen reference, fires, and it returns the predicted value.

A trained model exhibits this failure. The word-level alignment objective from the proposal's section 3.2.2, a squared difference between representations of dictionary-linked words in a sentence and its translation, is globally minimized when every representation is zero. Fine-tuning Qwen3-0.6B with it for 400 steps, three seeds, shrank mean representation norm from 451 to 42 and raw concept variance at the measured layer from 13,791 to 699, a factor of twenty. LFS moved from 0.445 to 0.493. MEXA and AaR ranked the collapsed model first.

The figure reads the five conditions two ways. Left, the ratio separates the collapsed condition from the sound cosine condition by 0.01. Right, raw concept variance separates them by a factor of twenty. The decomposition sees it; the ratio divides it out. That is why the paper reports LFS with its raw components.

A bound: the collapsed model matched its control on a 40-language Belebele panel, 0.395 versus 0.390, so collapse detection is a reorganization flag, not a harm measurement.
""")


def slide_12_tail(prs):
    s = new_slide(prs, "The tail: means hide failing sentences", 12)
    img_w = 7.5
    fit_picture(s, IMG["tail"], MARGIN, 1.5, img_w, 4.9)
    tx = MARGIN + img_w + 0.4
    add_text(s, tx, 1.6, SLIDE_W - MARGIN - tx, 5.0, [
        "AaR@q: mean retrieval margin of the worst q percent of sentences per language; "
        "13 models of the 1,500-sentence study, 1,500 sentences.",
        "A steep descent toward AaR@1 means rare, severe failures; a flat low curve means broad "
        "fragility.",
        {"runs": [("The tail measure tracks content transfer at ", {}),
                  ("0.443", {"bold": True, "color": ACCENT}), (" [0.09, 0.71].", {})]},
    ], size=BODY_PT, color=TEXT, space_after=16)
    add_notes(s, """
Averages across sentences hide the sentences that fail. The family's tail measure is Alignment-at-Risk: for each language, rank sentences by their retrieval margin, the gap between the correct translation's similarity and the strongest wrong candidate, and take the mean margin over the worst q percent. AaR@10 is the worst decile.

The figure shows the tail curves for 13 of the 14 models of the 1,500-sentence study at the dip layer, 1,500 sentences, averaged across the 128 languages; Salamandra-2B sits far below the axis and is omitted for legibility. Reading the curves: a steep descent toward AaR@1 means rare but severe failures, a catastrophic-tail profile; a flat low curve means broad fragility. Qwen3-8B has the healthiest tail throughout. OLMo-2-7B is ordinary at the worst 20 percent and falls steeply at the worst 1 percent. Two models with a similar mean can have very different tails, and that is what a mean hides.

The tail measure is validated on its own question. Against content transfer on the 19 frozen models it reaches 0.443 with model interval 0.09 to 0.71. Against pooled benchmark accuracy it reaches 0.105 after confound removal on the 33-model panel. Per-language estimates are noisy because the worst decile of 300 sentences is 30 sentences, so cross-language and cross-model aggregates are the reliable objects.

The point for the family: this is a different question from the share, answered by a different measure, and never averaged into it.
""")


def slide_13_family(prs):
    s = new_slide(prs, "A family, not a score", 13)
    rows = [
        ("Language vs meaning, by layer", "LFS profile"),
        ("Did concept variance collapse?", "Raw components"),
        ("Is the cross-language map simple?", "Nested maps"),
        ("Is the shared factor English?", "Hub test"),
        ("Do the hardest sentences align?", "Tail measure"),
    ]
    tbl_top, tbl_w = 1.5, 8.6
    row_h = 0.6
    gt = s.shapes.add_table(len(rows), 2, Inches(MARGIN), Inches(tbl_top), Inches(tbl_w),
                            Inches(row_h * len(rows)))
    table = gt.table
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    table.columns[0].width = Inches(5.4)
    table.columns[1].width = Inches(tbl_w - 5.4)
    for i, (q, r) in enumerate(rows):
        table.rows[i].height = Inches(row_h)
        for j, txt in enumerate((q, r)):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if j == 0 else LIGHT
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            run = p.add_run()
            run.text = txt
            set_run(run, size=BODY_PT, bold=(j == 1), color=(NAVY if j == 1 else TEXT))

    below = tbl_top + row_h * len(rows) + 0.35
    add_text(s, MARGIN, below, CONTENT_W, 0.6, [
        "MEXA: published reference measure, reported alongside, never ranked."
    ], size=MIN_BODY_PT, color=GRAY, italic=True, space_after=0)
    add_rect(s, MARGIN, below + 0.75, CONTENT_W, 0.9, LIGHT)
    add_text(s, MARGIN + 0.3, below + 0.75, CONTENT_W - 0.6, 0.9, [
        {"runs": [("In Aim 2: design, monitoring, evaluation. ", {"color": NAVY}),
                  ("Never the loss.", {"bold": True, "color": ACCENT})]},
    ], size=22, color=NAVY, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    add_notes(s, """
Here is the framing I would like you to approve. A family of measures, each answering one question, none averaged into a score. The proposal asks for metrics, plural; the paper delivers a family. This is not a leaderboard paper.

The five rows. The LFS profile and its dip depth answer how variation splits between language and meaning at each layer. The raw components, SS_lang, SS_con, the residual, and the norm against a frozen reference, answer whether scale or concept variance collapsed, which the ratio hides. The nested map sequence answers whether the cross-language map is simple: additive, rotational, or nonlinear. The hub test answers whether the shared factor is English. The tail measure answers whether the hardest sentences align.

MEXA is a published reference measure computed alongside. It asks whether a translation is the nearest neighbor against an English pivot; LFS asks how variance splits, with no pivot, a null, size stability, and components that catch what MEXA ranked first. Different questions, and the paper ranks nothing against it.

For Aim 2, the family's demonstrated role is to choose the layer, watch for collapse, and evaluate whether structure and behavior moved. Not the loss. Minimizing LFS as a training loss reversed the observational association, Pearson plus 0.92 between LFS and tail alignment across 18 condition-seed points, and a bounded variance-preservation penalty cannot restrain an unbounded alignment term. The constraint form, align subject to preserving raw concept variance and capability, is the Aim 2 design.
""")


def slide_14_asks(prs):
    s = new_slide(prs, "Four asks, two dates", 14)
    items = [
        "Framing and title: a family of measures, a structural diagnostic, not a leaderboard.",
        "Author list and order, final by 18 September.",
        "One registered reviewer for three papers, or the submission is desk-rejected.",
        "Green light to register the abstract.",
    ]
    top0, row_h = 1.6, 0.95
    for i, item in enumerate(items):
        y = top0 + i * row_h
        add_rect(s, MARGIN, y + 0.08, 0.62, 0.62, ACCENT, shape=MSO_SHAPE.OVAL)
        add_text(s, MARGIN, y + 0.08, 0.62, 0.62, [str(i + 1)], size=22, color=WHITE, bold=True,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0)
        add_text(s, MARGIN + 0.9, y, CONTENT_W - 0.9, row_h, [item], size=22, color=TEXT,
                 anchor=MSO_ANCHOR.MIDDLE, space_after=0)
    strip_top = 5.65
    add_rect(s, MARGIN, strip_top, CONTENT_W, 0.9, NAVY)
    add_text(s, MARGIN + 0.3, strip_top, CONTENT_W - 0.6, 0.9, [
        {"runs": [("Abstract 18 September AOE", {"bold": True}),
                  ("   ·   ", {}),
                  ("Full paper 25 September AOE", {"bold": True})]},
    ], size=22, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, space_after=0)
    add_notes(s, """
The asks, in order.

First, the framing and the title. A family of measures, a structural diagnostic reported with its raw components and behavioral checks, not a universal score and not a loss.

Second, the author list and order. The abstract registration deadline is 18 September, anywhere on earth, and the author list must be final by then. I would like to register the abstract as soon as you approve the list.

Third, a registered reviewer. ICLR requires an author on each submission to register and review three papers, or the submission is desk-rejected. One of you or Kenton would need to register.

Fourth, a green light to register the abstract with the approved list. The full paper is due 25 September, anywhere on earth. Both documents are attached; the draft is complete, and the remaining items are the anonymous code link, pinned revisions for the primary set, and the wording of the ethics and AI-use statements, which need your confirmation.

If we have time, the weakest point is the benchmark target, and I would rather discuss it now than have a reviewer raise it. Two independent benchmarks agree on only a third of their per-language residual variation, so the ceiling for any predictor is about 0.56; that is a finding about evaluation, not a hole in the metric.

Closing sentence: I would like to register the abstract by the 17th with the author list you approve. The full paper goes in on the 25th.
""")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    for fn in (slide_01_title, slide_02_question, slide_03_pipeline, slide_04_definition,
               slide_05_profiles, slide_06_depth, slide_07_stable, slide_08_budget,
               slide_09_content, slide_10_exam, slide_11_collapse, slide_12_tail,
               slide_13_family, slide_14_asks):
        fn(prs)
    return prs


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _shape_text(shape) -> str:
    parts = []
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for sh in shape.shapes:
            parts.append(_shape_text(sh))
        return " ".join(parts)
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        parts.append(shape.text_frame.text)
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            for cell in row.cells:
                parts.append(cell.text_frame.text)
    return " ".join(parts)


def _min_font_pt(shape) -> float | None:
    sizes = []

    def collect(tf):
        for p in tf.paragraphs:
            for r in p.runs:
                if r.font.size is not None and r.text.strip():
                    sizes.append(r.font.size.pt)

    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for sh in shape.shapes:
            m = _min_font_pt(sh)
            if m is not None:
                sizes.append(m)
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        collect(shape.text_frame)
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            for cell in row.cells:
                collect(cell.text_frame)
    return min(sizes) if sizes else None


def verify(path: Path) -> list[int]:
    prs = Presentation(str(path))
    assert len(prs.slides) == 14, f"expected 14 slides, found {len(prs.slides)}"
    counts = []
    problems = []
    for idx, slide in enumerate(prs.slides, start=1):
        words = 0
        pictures = 0
        all_text = []
        for shape in slide.shapes:
            t = _shape_text(shape)
            all_text.append(t)
            words += len(t.split())
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pictures += 1
            m = _min_font_pt(shape)
            # slide-number box is the only sub-18pt text allowed
            if m is not None and m < MIN_BODY_PT and t.strip() != str(idx):
                problems.append(f"slide {idx}: text below {MIN_BODY_PT} pt ({m} pt): {t[:40]!r}")
        counts.append(words)
        if words > MAX_WORDS:
            problems.append(f"slide {idx}: {words} visible words (> {MAX_WORDS})")
        notes = slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""
        n_notes = len(notes.split())
        if not (NOTES_MIN <= n_notes <= NOTES_MAX):
            problems.append(f"slide {idx}: notes {n_notes} words (want {NOTES_MIN} to {NOTES_MAX})")
        joined = (" ".join(all_text) + " " + notes).lower()
        for b in BANNED:
            if b in joined:
                problems.append(f"slide {idx}: banned wording {b!r}")
        expected_pics = {3: 1, 5: 1, 8: 1, 9: 1, 11: 1, 12: 1}.get(idx, 0)
        if pictures != expected_pics:
            problems.append(f"slide {idx}: {pictures} pictures, expected {expected_pics}")
    for name, p in IMG.items():
        if not p.exists():
            problems.append(f"missing image {name}: {p}")
    if problems:
        raise AssertionError("\n".join(problems))
    return counts


def main() -> int:
    for name, p in IMG.items():
        assert p.exists(), f"missing image {name}: {p}"
    prs = build()
    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(out))
    results = {}
    for out in OUT_PATHS:
        results[str(out)] = verify(out)
    counts = list(results.values())[0]
    for other in results.values():
        assert other == counts, "word counts differ between the two saved copies"
    print("OUTPUTS")
    for out in OUT_PATHS:
        print(f"  {out}  ({out.stat().st_size} bytes)")
    print("VISIBLE WORDS PER SLIDE (max 60, slide number included)")
    for i, c in enumerate(counts, start=1):
        print(f"  slide {i:2d}: {c}")
    print(f"  max = {max(counts)}, all <= {MAX_WORDS}: {all(c <= MAX_WORDS for c in counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
