#!/usr/bin/env python3
"""Visual companion to the ICLR draft for Professor Koehn's pre-submission review.
Core: title; how LFS works (3); evidence (3); limits (1); what needs feedback (1). Appendix: detailed slides
reused from the reading deck. Layout and figures shared with make_reading_deck.py."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import make_reading_deck as R
from make_reading_deck import (textbox, fit, W, H, NAVY, INK, GREY, ACCENT, RULE, TINT, BANNED, FONT_PATH,
                               IMG_X, IMG_Y, IMG_W, IMG_H, TXT_X, TXT_W, PARA_Y, PARA_H, NUM_Y, NUM_H, PL, RD, IF)
OUT = [R.ROOT / "paper" / "slides_pptx" / "Waiz_Khan_LFS_Companion_2026-09-15.pptx",
       R.ROOT / "deliverables" / "2026-09-15" / "Waiz_Khan_LFS_Companion_2026-09-15.pptx"]

S = []
def slide(title, image=None, text="", numbers=None, images=None, cards=None, bullets=None):
    S.append(dict(title=title, image=image, text=text, numbers=numbers or [], images=images, cards=cards, bullets=bullets))

slide("Measuring the Language Share of Multilingual Representations", bullets=["Waiz Khan"])

slide("How LFS works, 1 of 3: from parallel sentences to a grid of vectors", PL / "pl2_grid.png",
      "The same 300 NTREX sentences in 128 languages are run once through a frozen base model. At every layer the hidden states of a text are mean-pooled in fp32 into one vector, so each layer gives a grid: one row per language, one column per sentence. Each coordinate is then standardized jointly over the whole grid, never per language, because per-language standardization would remove the language differences being measured.",
      ["L = 128 languages, N = 300 sentences, 38,400 cells per layer", "d = 4,096 for Qwen3-8B; K = 36 blocks, so 37 layers", "joint standardization matters: Qwen3-8B, layer 19, reads 0.347 raw and 0.608 standardized"])

slide("How LFS works, 2 of 3: the decomposition and the share", PL / "pl5_sums_of_squares.png",
      "The total spread of the standardized grid splits exactly into a language part (how far the language means sit from the grand mean), a sentence part (the same for sentence means), and a residual (interaction plus noise). LFS is the language part as a share of the two systematic parts; the residual is excluded on purpose. Repeating this at every layer gives the depth profile, summarized by the minimum layer, the minimum share, and the dip depth.",
      ["LFS = SS_lang / (SS_lang + SS_con), defined when the sum is positive", "expected value under pure noise: (L − 1)/(L + N − 2) = 0.298 on this grid", "dip depth = LFS at layer 0 minus the minimum LFS"])

slide("How LFS works, 3 of 3: a numerical example", PL / "pl8_worked_example.png",
      "On a two-language, three-sentence grid with one number per cell the arithmetic is visible: language means 6 and 9, sentence means 3.5, 7.5, 11.5, grand mean 7.5; the language part is 13.5, the sentence part 64, the residual 0, so LFS is 0.17. The real computation is the same over 4,096 coordinates. For Qwen3-8B at layer 19 the language part is 55.2 million and the sentence part 35.6 million, giving 0.608, against 0.945 at layer 0 and 0.958 at the last layer.",
      ["tiny grid: SS_lang 13.5, SS_con 64, SS_res 0, LFS 0.17", "Qwen3-8B, layer 19: 55.2M / (55.2M + 35.6M) = 0.608", "profile: 0.945 at layer 0, 0.608 at layer 19, 0.958 at layer 36; dip depth 0.336"])

slide("Evidence 1: every profile has an interior minimum, and it replicates", IF / "fig1_lfs_profiles_17.png",
      "All 17 primary models (direct sums of squares) and all 19 expansion models (the variance-component estimator, LFS-VC) have a profile that is high at the input, lowest at an interior layer, and high again at the output. Dip depth ranges nine-fold and is set by training recipe more than by size. The profile replicates on FLORES-200, is stable under sentence resampling, survives instruction tuning, and held on two model families whose predictions were frozen before the run.",
      ["dip depth 0.039 (BLOOM-1.7B) to 0.336 (Qwen3-8B); 36 of 36 profiles", "FLORES-200: 17 of 17, depth ordering Spearman 0.92", "resampling: half-widths 0.004 to 0.020 for 12 of 14 models; ordering Spearman ≥ 0.90 in 2,000 of 2,000", "instruction tuning: six pairs, dip depth within 0.01; unseen families: 4 of 4 predictions held"])

slide("Evidence 2: the shared middle is multilingual, and the map between languages is simple", RD / "r7_evidence2.png",
      text="Left: at each model's dip layer, the average of the other languages predicts a language's vectors better than English does for 113 to 124 of 127 languages, so the shared middle is not English. Right: mapping each language into a shared reference space with nested maps, a per-language offset plus a scale removes 63 to 97 percent of the held-out cross-language misalignment; rotation never adds more than 0.7 percent.",
      numbers=["hub test: 17 models, ridge on 64 principal components, 3-fold cross-validation; median gain +0.06 to +0.16 R²", "decomposition: 14 models, 1,500 sentences, common rank 64; offset + scale 63% to 97%", "rotation at most 0.7%; the difference between languages is mostly an offset"])

slide("Evidence 3: the sentence-component share and the use of foreign-language context", IF / "fig_r1_model_level.png",
      "Under a protocol frozen before the run, 19 models from 13 families were scored on whether a preceding sentence in another language (41 languages) lowers the loss on the English sentence that follows, against a length-matched mismatched context. Models with a smaller language share at the dip layer, one minus LFS-VC, use the foreign context more. The association holds after adjusting for training-data exposure, language family, script, and tokenizer fertility. The same share shows no clear association with pooled benchmark accuracy, which the paper reports as a limit.",
      ["Spearman 0.77 [0.44, 0.94] unadjusted; 0.51 [0.19, 0.71] after covariate adjustment", "positive under leave-one-family-out; 157,700 scored sequences", "pooled Belebele and INCLUDE accuracy, 33 models: 0.04 [−0.05, 0.12]; MEXA 0.24 for reference"])

slide("What limits the claims: uniform scaling and training interventions", IF / "fig17_ratio_hides_scale.png",
      "A share cannot see uniform scaling. A word-alignment fine-tuning of Qwen3-0.6B shrank the representation norm from 451 to 42 and the raw content variance twenty-fold while LFS moved only from 0.445 to 0.493 and the retrieval score rose; only the raw component caught it. And LFS is not a training objective: across 28 conditions at three seeds, objectives that pushed LFS down moved tail alignment down with it, and training on LFS itself doubled the dip while Belebele fell. The paper therefore presents LFS as a diagnostic reported with its raw components, not as a score or a loss.",
      ["norm 451 → 42; raw content variance 13,791 → 699; LFS 0.445 → 0.493; MEXA +0.09 over the control", "Belebele unchanged for the shrunk model (0.395 vs 0.390): representation change, not demonstrated harm", "training interventions: Pearson +0.92 over 18 condition-seed points; LFS as a loss: dip doubled, Belebele −2.9 vs −1.7 for the control"])

slide("What I need your feedback on before submission", cards=[
      ("Contribution", "The paper leads with LFS as a structural diagnostic: a two-way analysis-of-variance share with a known noise value, profiles on 33 models, one behavioral association, and explicit limits. Is that the right claim to lead with, or should the behavioral result carry more weight?"),
      ("Framing", "It is presented as a measurement paper, not a method that improves models. The English question is phrased as \"a multilingual average predicts better than English\", MEXA is a reference measure and is never ranked against, and the limits get a full section. Does that read as careful or as weak?"),
      ("Decisions and gaps", "Title: current choice or one of the two alternatives. Author order and the reciprocal reviewer by 18 September; paper by 25 September. Open evidence: the 19 expansion models are measured with LFS-VC only; a direct-SS re-measurement needs cluster time. The anonymous code link is added at submission.")])

# ---- appendix: detailed slides reused from the reading deck
want = ["The problem", "What the proposal asks (Aim 1)", "Step 1: from a sentence to one vector per layer", "Step 3: put every coordinate on the same scale",
        "Step 4: three kinds of mean", "Step 6: the Language-Factor Share", "Step 7: repeat at every layer and read the curve",
        "Experiment 1: 17 models, every profile has an interior minimum", "Experiment 2: does size decide the dip?",
        "Experiment 5: does the model use context given in another language?", "Experiment 6: two behaviors, the same three measures",
        "How the measurement was checked", "What this answers in the proposal", "The reporting rule and the limits", "What Aim 2 gets from this"]
by_title = {s["title"]: s for s in R.SLIDES}
for w in want:
    s = dict(by_title[w]); s["title"] = "Appendix: " + s["title"][0].lower() + s["title"][1:] if not s["title"].startswith(("Step", "Experiment")) else "Appendix, " + s["title"]
    s.update(images=None, cards=None); S.append(s)

def place_image(sl, p, x, y, bw, bh):
    p = Path(p); assert p.exists(), f"missing image {p}"
    w, h = fit(p, bw, bh); yy = y + (bh - h) / 2 if h < bh * 0.6 else y
    sl.shapes.add_picture(str(p), Inches(x + (bw - w) / 2), Inches(yy), Inches(w), Inches(h))

def build():
    prs = Presentation(); prs.slide_width = Inches(W); prs.slide_height = Inches(H); blank = prs.slide_layouts[6]
    for i, s in enumerate(S, 1):
        sl = prs.slides.add_slide(blank)
        if i == 1:
            textbox(sl, 1.2, 2.3, W - 2.4, 2.0, [s["title"]], 36, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
            rule = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.25), Inches(4.45), Inches(1.4), Inches(0.06)); rule.fill.solid(); rule.fill.fore_color.rgb = ACCENT; rule.line.fill.background()
            textbox(sl, 1.2, 6.0, W - 2.4, 0.6, [s["bullets"][0]], 20, color=INK); continue
        textbox(sl, 0.5, 0.3, W - 1.0, 0.75, [s["title"]], 24 if len(s["title"]) < 66 else 19, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        rule = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.56), Inches(1.08), Inches(1.1), Inches(0.05)); rule.fill.solid(); rule.fill.fore_color.rgb = ACCENT; rule.line.fill.background()
        if s.get("cards"):
            n = len(s["cards"]); cw = (W - 1.0 - 0.3 * (n - 1)) / n
            for k, (head, body) in enumerate(s["cards"]):
                x = 0.5 + k * (cw + 0.3)
                box = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.45), Inches(cw), Inches(5.3)); box.fill.solid(); box.fill.fore_color.rgb = TINT; box.line.fill.background()
                bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.45), Inches(cw), Inches(0.06)); bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
                textbox(sl, x + 0.2, 1.65, cw - 0.4, 0.5, [head], 16, bold=True, color=NAVY)
                textbox(sl, x + 0.2, 2.2, cw - 0.4, 4.4, [body], 13, fit_max=13)
        else:
            if s.get("images"):
                half = (IMG_W - 0.2) / 2
                place_image(sl, s["images"][0], IMG_X, IMG_Y, half, IMG_H); place_image(sl, s["images"][1], IMG_X + half + 0.2, IMG_Y, half, IMG_H)
            elif s["image"]: place_image(sl, s["image"], IMG_X, IMG_Y, IMG_W, IMG_H)
            div = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.72), Inches(1.35), Inches(0.012), Inches(5.55)); div.fill.solid(); div.fill.fore_color.rgb = RULE; div.line.fill.background()
            if s["text"]:
                textbox(sl, TXT_X, PARA_Y, TXT_W, PARA_H, [s["text"]], 12, fit_max=12)
                box = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(TXT_X), Inches(NUM_Y), Inches(TXT_W), Inches(NUM_H)); box.fill.solid(); box.fill.fore_color.rgb = TINT; box.line.fill.background()
                bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(TXT_X), Inches(NUM_Y), Inches(0.06), Inches(NUM_H)); bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
                textbox(sl, TXT_X + 0.15, NUM_Y + 0.08, TXT_W - 0.25, 0.32, ["Key numbers"], 10.5, bold=True, color=NAVY)
                textbox(sl, TXT_X + 0.15, NUM_Y + 0.42, TXT_W - 0.25, NUM_H - 0.5, ["•  " + n for n in s["numbers"]], 10.5, fit_max=10.5, space_after=3)
        textbox(sl, 0.5, 7.08, 8, 0.3, ["Measuring the Language Share of Multilingual Representations"], 9, color=GREY)
        textbox(sl, W - 1.6, 7.08, 1.1, 0.3, [f"{i} / {len(S)}"], 9, color=GREY, align=PP_ALIGN.RIGHT)
        sl.notes_slide.notes_text_frame.text = s["text"] or " ".join(b for _, b in (s.get("cards") or []))
    return prs

def check(path):
    prs = Presentation(str(path)); problems = []
    for i, sl in enumerate(prs.slides, 1):
        vis = " ".join(sh.text_frame.text for sh in sl.shapes if sh.has_text_frame); low = vis.lower()
        for b in BANNED:
            if b in low: problems.append(f"slide {i}: banned word {b!r}")
        print(f"  slide {i:2d}: {len(vis.split()):3d} words")
    if problems: raise AssertionError("\n".join(problems))

if __name__ == "__main__":
    prs = build()
    for out in OUT: out.parent.mkdir(parents=True, exist_ok=True); prs.save(str(out)); print("wrote", out)
    check(OUT[0]); print("checks passed:", len(S), "slides")
