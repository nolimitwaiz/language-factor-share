#!/usr/bin/env python3
"""Reading deck: LFS from the proposal to the experiments, written to be read without a presenter.
Layout per slide: navy title band, figure on the left, explanation and a key-numbers box on the right.
Every number is taken from the paper, the technical report, or the stored results."""
from pathlib import Path
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "docs" / "study" / "figs"; PL = F / "pipeline"; RD = F / "reading"; IF = ROOT / "paper" / "iclr2027" / "figs"
OUT = [ROOT / "paper" / "slides_pptx" / "Waiz_Khan_LFS_Reading_Deck_2026-09-13.pptx",
       ROOT / "deliverables" / "2026-09-13" / "Waiz_Khan_LFS_Reading_Deck_2026-09-13.pptx"]
NAVY = RGBColor(0x0B, 0x2A, 0x4A); WHITE = RGBColor(0xFF, 0xFF, 0xFF); GREY = RGBColor(0x66, 0x66, 0x66)
BOX = RGBColor(0xEE, 0xF3, 0xF9); BOXLINE = RGBColor(0xC9, 0xD6, 0xE6); INK = RGBColor(0x1A, 0x1A, 0x1A); ORANGE = RGBColor(0xE4, 0x57, 0x2E)
W, H = 13.333, 7.5
BANNED = ["fingerprint", "toy", "battery", "tripwire", "u-shaped", "destroyed", "baseline", "beats", "first of its kind", "campaign", "cohort",
          "walk-forward", "headline", "deflat", "ladder", "harness", "concept direction", "confounds removed", "meaning-weighted", "language-agnostic",
          "ranked first", "scored best", "held-out famil", "—"]
FONT_PATH = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibri.ttf"

SLIDES = []
def slide(title, image=None, text="", numbers=None, bullets=None):
    SLIDES.append(dict(title=title, image=image, text=text, numbers=numbers or [], bullets=bullets))

# ---------------------------------------------------------------- content
slide("Measuring the Language Share of Multilingual Representations", None, "", bullets=["Waiz Khan"])

slide("The problem", PL / "c1_problem.png",
      "Give one model the same reading test in ten languages and the scores fall from 0.95 in English to 0.48 in Amharic. These are real zero-shot Belebele accuracies for Qwen3-8B. Aim 1 of the proposal asks what happens inside the model before anyone tries to fix this. The model is a block of numbers nobody wrote by hand, so the first step is a measurement that shows where the languages part ways.",
      ["Qwen3-8B, zero-shot Belebele accuracy, ten languages", "English 0.953, German 0.887, French 0.903, Spanish 0.890", "Russian 0.833, Arabic 0.870, Persian 0.833", "Swahili 0.573, Khmer 0.607, Amharic 0.477"])

slide("What the proposal asks (Aim 1)", PL / "c2_proposal.png",
      "Section 3.1.1 of the proposal pictures three stages: take in the word, process the content, pick the output word. The first and last should depend on the language; the middle should be shared. The proposal then states the open question this project answers: it is not clear how strong the language component in the representation is, or should be. It also asks whether the spaces of different languages can be joined by a simple map, as older word-vector spaces could.",
      ["Question 1: how strong is the language component, layer by layer?", "Question 2: is English privileged in the middle layers?", "Question 3: how simple is the map between languages?", "Question 4: does any of this show in behavior?"])

slide("Step 1: from a sentence to one vector per layer", PL / "pl1_sentence_to_vectors.png",
      "A sentence is split into tokens. Each token starts as a vector, and every Transformer block rewrites all of them; layer 0 is the embedding output and layer K the last block. At each layer we take the hidden states, one row per token, and average them over the tokens in full precision. That gives one vector per sentence per layer. Nothing is generated; we only read the states of frozen base models.",
      ["pooled vector: h̄^(k) = (1/T) Σ_t h_t^(k)", "d = 4,096 for Qwen3-8B; K = 36 blocks, so 37 layers", "fp32 pooling; base (not instruction-tuned) models"])

slide("Step 2: the language-by-sentence grid", PL / "pl2_grid.png",
      "The data are 300 news sentences translated by professionals into 128 languages (NTREX-128). Every text runs once through the frozen model. Arranged by language and sentence, the vectors form a grid: one row per language, one column per sentence, one grid per layer. Down a column is the same sentence in 128 languages; across a row is one language telling 300 different sentences.",
      ["L = 128 languages, N = 300 sentences, 38,400 cells", "cell (ℓ, c) holds the pooled vector h̄_ℓc^(k) in R^d", "one grid per layer k = 0 … K"])

slide("Step 3: put every coordinate on the same scale", PL / "pl3_standardize.png",
      "Before any arithmetic, every coordinate is standardized over the whole grid: subtract its mean and divide by its standard deviation, both computed over all 38,400 cells. This is done jointly across languages, never per language, because per-language standardization would remove exactly the language differences we want to measure. It matters: in Qwen3-8B one coordinate, number 2276, carries 92.8 percent of the raw spread at layer 19.",
      ["z_ℓcd = (h_ℓcd − mean_d) / s_d", "without this step the share reads 0.347; with it, 0.608 (Qwen3-8B, layer 19)", "after standardization coordinate 2276 carries 0.02 percent of the language part"])

slide("Step 4: three kinds of mean", PL / "pl4_means.png",
      "From the standardized grid we take three averages. Averaging across a row gives the language mean: what that language looks like on average. Averaging down a column gives the sentence mean: what that sentence looks like across languages. Averaging every cell gives the grand mean. Everything that follows is built from how far these means sit from each other.",
      ["language mean: z̄_ℓ = (1/N) Σ_c z_ℓc", "sentence mean: z̄_c = (1/L) Σ_ℓ z_ℓc", "grand mean: z̄ = (1/LN) Σ_ℓc z_ℓc"])

slide("Step 5: split the total spread into three parts", PL / "pl5_sums_of_squares.png",
      "The total spread, the squared distance of every cell from the grand mean, splits exactly into three parts. The language part is how far the language means sit from the grand mean, counted N times because each language mean stands for N cells. The sentence part is the same for sentence means, counted L times. The residual is what is specific to one language-sentence pair: interaction plus noise. On a balanced grid the three add up to the total.",
      ["SS_lang = N Σ_ℓ ‖z̄_ℓ − z̄‖²", "SS_con = L Σ_c ‖z̄_c − z̄‖²", "SS_res = Σ_ℓc ‖z_ℓc − z̄_ℓ − z̄_c + z̄‖²", "SS_lang + SS_con + SS_res = total"])

slide("Step 6: the Language-Factor Share", PL / "pl6_share.png",
      "The Language-Factor Share is the language part divided by the language part plus the sentence part. The residual is left out on purpose, so a noisy model does not look language-neutral only because its residual is large. LFS is 1 when only language separates the cells and 0 when only the sentence does. Under pure noise it is not zero but 0.298 on this grid, because averages of random numbers still differ by chance; every real value is read against that number.",
      ["LFS = SS_lang / (SS_lang + SS_con), defined when the sum is positive", "noise expectation (L − 1)/(L + N − 2) = 127/426 = 0.298", "Qwen3-8B, layer 19: SS_lang 55.2 million, SS_con 35.6 million, residual 66.5 million (left out)", "LFS = 55.2 / (55.2 + 35.6) = 0.608; as shares of the total: 0.351, 0.226, 0.423"])

slide("Step 8: the same arithmetic on a two-by-three grid", PL / "pl8_worked_example.png",
      "To see the arithmetic, shrink the grid to two languages and three sentences with one number per cell. The language means are 6 and 9, the sentence means 3.5, 7.5 and 11.5, the grand mean 7.5. The language part is three times the two squared distances 2.25 and 2.25, which is 13.5. The sentence part is two times 16 plus 0 plus 16, which is 64. The residual is zero because this grid is exactly additive. The share is 13.5 over 77.5, which is 0.17. The real computation is this, 4,096 coordinates at a time.",
      ["SS_lang = 13.5, SS_con = 64, SS_res = 0", "LFS = 13.5 / 77.5 = 0.17", "noise value for a 2 × 3 grid: (2 − 1)/(2 + 3 − 2) = 1/3"])

slide("Step 7: repeat at every layer and read the curve", RD / "r1_profiles.png",
      "The computation is repeated at every layer, giving one share per layer: the depth profile. At the input the vectors are still the words, so language decides where they land and the share is near 1. In the middle the model has worked out what is being said, so the same sentence lands close together across languages and the share falls. At the output the model must produce a word in the right language, so the share rises again. Three numbers summarize a profile: the minimum layer, the minimum share, and the dip depth, the layer-0 value minus the minimum.",
      ["Qwen3-8B: 0.945 at layer 0, 0.608 at layer 19, 0.958 at layer 36; dip depth 0.336", "band: the range of 17 models; every one has this shape", "expected value under pure noise: 0.298; language never fully leaves"])

slide("Experiment 1: 17 models, every profile has an interior minimum", RD / "r2_dip_depths.png",
      "We ran the measurement on 17 open base models from 11 families. All 17 profiles have an interior minimum with both ends higher. What differs is how deep: from 0.039 for BLOOM-1.7B to 0.336 for Qwen3-8B, a nine-fold range. To see how much a dip depth depends on which 300 sentences were used, we resampled 2,000 sets of 300 from 1,500 stored sentences. The bars are the resulting 95 percent intervals: narrow for 12 of 14 models, wide only for the two whose minimum sits at an early layer.",
      ["17 of 17 dip and recover; 19 of 19 in the 19-model expansion set", "dip depth 0.039 (BLOOM-1.7B) to 0.336 (Qwen3-8B)", "interval half-widths 0.004 to 0.020 for 12 models; 0.045 (Mistral-7B) and 0.059 (Salamandra-2B)", "model ordering agrees with the full grid at Spearman ≥ 0.90 in 2,000 of 2,000 resamples"])

slide("Experiment 2: does size decide the dip?", RD / "r3_size_vs_family.png",
      "Within one family the dip deepens with size: Qwen3 goes 0.206, 0.213, 0.314, 0.336 from 0.6B to 8B. But families at the same size differ far more than sizes within a family. Qwen3-1.7B dips deeper than every 7B model from another lab except Llama-3.1-8B; BLOOM, trained on a balanced 46-language mix, barely dips at either size; Salamandra dips more at 2B than at 7B. So dip depth is set by how a model was trained more than by how big it is. This is a descriptive comparison: we do not separate data, tokenizer, and training recipe.",
      ["Qwen3: 0.206 → 0.213 → 0.314 → 0.336", "BLOOM: 0.039 (1.7B) and 0.043 (7.1B)", "Salamandra: 0.254 (2B) versus 0.180 (7B)", "nine-fold range across families at similar sizes"])

slide("Experiment 3: is the shared middle English?", RD / "r4_hub_wins.png",
      "Some papers say models think in English. We tested this at each model's dip layer. For each of the 127 non-English languages we fit two predictors of its vectors from the same sentences: the English vectors, and the average of all the other languages' vectors. Each predictor is scored on sentences it never saw. The average wins for 113 to 124 of 127 languages, depending on the model, by a median of 0.06 to 0.16 R². English is the best single donor only for high-resource targets. The shared middle is multilingual, not English.",
      ["ridge regression on 64 principal components, 3-fold cross-validation", "average of the other languages wins for 113 (BLOOM-1.7B) to 124 of 127 languages", "median gain +0.06 to +0.16 R² per model", "17 primary models, dip layer of each"])

slide("Experiment 4: how complicated is the difference between languages?", IF / "fig15_budget_composition.png",
      "At the dip layer the languages still sit apart. How complicated is the difference? We map each language's vectors into a shared reference space (a generalized-Procrustes consensus) with a nested sequence of maps: a per-language offset, then a scale, then a rotation, a general linear map, and a nonlinear map. Each map is credited only with the error it removes on held-out sentences. The offset plus the scale remove 63 to 97 percent of the misalignment; rotation never adds more than 0.7 percent. The map between languages is simple.",
      ["14 models, 1,500 sentences per language, common rank 64", "offset + scale: 63% (Qwen3-4B, Qwen3-0.6B) to 97% (OLMo-2-1B, OLMo-2-7B)", "Qwen3-8B: offset 36% + scale 28% = 64%", "rotation at most 0.7% (BLOOM-7.1B)"])

slide("Experiment 5: does the model use context given in another language?", RD / "r6_context_test.png",
      "The proposal's third question is whether information given in one language is used in another. We took an English sentence and the sentence before it from the same news story, translated into 41 languages. We measured the model's loss on the English sentence with the true foreign-language context, and with a context of the same language and length taken from a different story. The difference is the benefit of the context, normalized by the same benefit with English context. The mismatched control matters, because foreign text from the wrong story hurts.",
      ["100 sentence pairs from 100 documents; 41 context languages", "19 models from 13 families; 157,700 scored sequences", "teacher-forced fp32 loss on the target tokens only", "protocol and predictions frozen before the run"])

slide("Experiment 5, result: a smaller language share goes with more use of foreign context", IF / "fig_r1_model_level.png",
      "Each point is one model, placed by its sentence-component share at the dip layer (one minus LFS-VC, the variance-component estimator used on this set) and its median benefit over 34 languages. Models with a smaller language share use the foreign-language context more. The raw model-level correlation is 0.77; after adjusting for training-data exposure, language family, script, and tokenizer fertility it is 0.51, and it stays positive when any single family is removed.",
      ["Spearman 0.77 [0.44, 0.94] unadjusted, 19 models", "0.51 [0.19, 0.71] after covariate adjustment (model bootstrap)", "positive under leave-one-family-out", "MEXA on the same test: 0.59 [0.33, 0.82], reported for reference"])

slide("Experiment 6: two behaviors, the same three measures", RD / "r5_two_behaviors.png",
      "The same three measures tested against two behaviors. Left: pooled multiple-choice accuracy (Belebele and INCLUDE) on 33 models, 70 languages, 1,740 rows after covariate filtering. No measure predicts it well, LFS least of all. Right: the foreign-context benefit on 19 models, where all three are clearly associated. Part of the reason is the target: two benchmarks on the same cells agree on only about a third of their adjusted variation. We report the null result as a limit, not a defect to tune away.",
      ["pooled accuracy: 1 − LFS-VC 0.04 [−0.05, 0.12]; MEXA 0.24; AaR 0.10", "foreign-context benefit: 0.51, 0.59, 0.44", "two benchmarks agree on 0.315 of their adjusted residual variation"])

slide("Experiment 7: a share cannot see shrinking", IF / "fig17_ratio_hides_scale.png",
      "Multiply every vector by the same constant and LFS, MEXA, and CKA all stay put: a share cannot see uniform shrinking. This is not hypothetical. Fine-tuning Qwen3-0.6B with a word-alignment objective from the literature shrank the representation norm from 451 to 42 and the raw content variance from 13,791 to 699, twenty-fold. LFS moved only from 0.445 to 0.493, and the retrieval score rose. Only the raw content variance caught it. That is why every LFS value is reported with its raw components.",
      ["norm 451 → 42; raw content variance 13,791 → 699", "LFS 0.445 → 0.493; MEXA +0.09 over the LM-only control", "Belebele accuracy unchanged: a change in representation, not demonstrated harm", "three seeds, 400 steps"])

slide("How the measurement was checked", PL / "c7_checks.png",
      "Before trusting the measurement we tried to break it five ways, each with predictions written down first. A second corpus, FLORES-200 with Wikipedia sentences: 17 of 17 profiles dip and recover, depth ordering Spearman 0.92. Resampled sentences: the intervals of Experiment 1, ordering preserved in every draw. Sealed predictions for two unseen families, SmolLM2 and Falcon3: 4 of 4 held. Injected faults: shifts and rotations recovered, uniform shrinking invisible to any ratio. Instruction tuning: six base and instruct pairs move dip depth by at most 0.01.",
      ["FLORES-200: 17 of 17; Spearman 0.92; median |Δ depth| 0.011; granite the outlier (0.190 → 0.074)", "one frozen prediction of twelve failed: the precision bound (Mistral-7B 0.045, Salamandra-2B 0.059 above 0.03)", "SmolLM2 0.068 (predicted ≤ 0.15); Falcon3 0.140 (predicted intermediate)"])

slide("What this answers in the proposal", PL / "c4_answers.png",
      "Back to the proposal's questions. How strong is the language component, layer by layer? Large everywhere and smallest at an interior layer, with the same shape in all 36 measured profiles from 33 models. Is the shared middle English? No: the multilingual average predicts 113 to 124 of 127 languages better than English does. Is the map between languages simple? Yes: an offset plus a scale removes 63 to 97 percent of the misalignment. Does it show in behavior? Yes for foreign-language context use; no for pooled test scores.",
      ["profiles: 36 of 36 measured (33 models)", "English privileged: no, 113 to 124 of 127", "map: offset + scale, 63 to 97 percent", "behavior: 0.51 for context use; 0.04 for pooled scores"])

slide("The reporting rule and the limits", PL / "c5_limits.png",
      "Two limits, found on purpose. A share is blind to size, so the raw components are always printed beside it. And LFS is not a training objective: across 28 training conditions at three seeds on one small model, objectives that pushed LFS down moved tail alignment down with it, and training on LFS itself doubled the dip while Belebele fell 2.9 points. Its place in the next aim is design, monitoring, and evaluation, not the loss.",
      ["28 conditions × 3 seeds on Qwen3-0.6B", "Pearson +0.92 between LFS and tail alignment over 18 condition-seed points (six conditions)", "LFS as a loss: dip doubled in 400 steps, Belebele −2.9 points"])

slide("What Aim 2 gets from this", PL / "c6_next_aim.png",
      "What this hands to Aim 2. Where to act: the interior layer with the smallest language share, layer 19 for Qwen3-8B and layer 2 for Mistral-7B. What to act on: the per-language offset, which carries most of the language leftover; removing it from saved states deleted 97 percent of the language part in a pre-test that needed no training. How to check a fix: print the raw components and run the behavioral test, never the share alone.",
      ["dip layers: Qwen3-8B 19 of 36, Mistral-7B 2 of 32", "offset removal: 97 percent of SS_lang gone, offline, no GPU", "checks: SS_lang, SS_con, residual, norm, and a behavioral test"])

# ---------------------------------------------------------------- rendering
from pptx.enum.text import MSO_AUTO_SIZE
def fit(img_path, box_w, box_h):
    with Image.open(img_path) as im: w, h = im.size
    s = min(box_w / w, box_h / h); return w * s, h * s
def textbox(sl, x, y, w, h, paras, size, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, fit_max=None, space_after=0):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.06); tf.margin_top = tf.margin_bottom = Inches(0.04)
    for k, text in enumerate(paras):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph(); p.alignment = align; p.space_after = Pt(space_after)
        r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Calibri"
    if fit_max:
        try: tf.fit_text(font_family="Calibri", max_size=fit_max, font_file=FONT_PATH)
        except Exception as e: print("fit_text failed:", e)
        for p in tf.paragraphs:
            for r in p.runs: r.font.color.rgb = color; r.font.bold = bold
    return tb

ACCENT = RGBColor(0xE4, 0x57, 0x2E); RULE = RGBColor(0xD9, 0xD9, 0xD9); TINT = RGBColor(0xF3, 0xF6, 0xFA)
IMG_X, IMG_Y, IMG_W, IMG_H = 0.5, 1.3, 8.0, 5.5
TXT_X, TXT_W = 9.0, 3.85
PARA_Y, PARA_H = 1.3, 3.25
NUM_Y, NUM_H = 4.7, 2.2

def build():
    prs = Presentation(); prs.slide_width = Inches(W); prs.slide_height = Inches(H); blank = prs.slide_layouts[6]
    for i, s in enumerate(SLIDES, 1):
        sl = prs.slides.add_slide(blank)
        if i == 1:
            textbox(sl, 1.2, 2.3, W - 2.4, 2.0, [s["title"]], 36, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
            rule = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.25), Inches(4.45), Inches(1.4), Inches(0.06)); rule.fill.solid(); rule.fill.fore_color.rgb = ACCENT; rule.line.fill.background()
            textbox(sl, 1.2, 6.0, W - 2.4, 0.6, [s["bullets"][0]], 20, color=INK)
            continue
        textbox(sl, 0.5, 0.3, W - 1.0, 0.75, [s["title"]], 26 if len(s["title"]) < 62 else 21, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
        rule = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.56), Inches(1.08), Inches(1.1), Inches(0.05)); rule.fill.solid(); rule.fill.fore_color.rgb = ACCENT; rule.line.fill.background()
        if s["image"]:
            p = Path(s["image"]); assert p.exists(), f"missing image {p}"
            w, h = fit(p, IMG_W, IMG_H); y = IMG_Y + (IMG_H - h) / 2 if h < IMG_H * 0.6 else IMG_Y
            sl.shapes.add_picture(str(p), Inches(IMG_X + (IMG_W - w) / 2), Inches(y), Inches(w), Inches(h))
        div = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.72), Inches(1.35), Inches(0.012), Inches(5.55)); div.fill.solid(); div.fill.fore_color.rgb = RULE; div.line.fill.background()
        if s["text"]:
            textbox(sl, TXT_X, PARA_Y, TXT_W, PARA_H, [s["text"]], 12, fit_max=12)
            box = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(TXT_X), Inches(NUM_Y), Inches(TXT_W), Inches(NUM_H)); box.fill.solid(); box.fill.fore_color.rgb = TINT; box.line.fill.background()
            bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(TXT_X), Inches(NUM_Y), Inches(0.06), Inches(NUM_H)); bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
            textbox(sl, TXT_X + 0.15, NUM_Y + 0.08, TXT_W - 0.25, 0.32, ["Key numbers"], 10.5, bold=True, color=NAVY)
            textbox(sl, TXT_X + 0.15, NUM_Y + 0.42, TXT_W - 0.25, NUM_H - 0.5, ["•  " + n for n in s["numbers"]], 10.5, fit_max=10.5, space_after=3)
        textbox(sl, 0.5, 7.08, 8, 0.3, ["Measuring the Language Share of Multilingual Representations"], 9, color=GREY)
        textbox(sl, W - 1.6, 7.08, 1.1, 0.3, [f"{i} / {len(SLIDES)}"], 9, color=GREY, align=PP_ALIGN.RIGHT)
        sl.notes_slide.notes_text_frame.text = s["text"]
    return prs

def check(path):
    prs = Presentation(str(path)); problems = []
    for i, sl in enumerate(prs.slides, 1):
        vis = " ".join(sh.text_frame.text for sh in sl.shapes if sh.has_text_frame); low = vis.lower()
        for b in BANNED:
            if b in low: problems.append(f"slide {i}: banned word {b!r}")
        sizes = sorted({r.font.size.pt for sh in sl.shapes if sh.has_text_frame for p in sh.text_frame.paragraphs for r in p.runs if r.font.size})
        print(f"  slide {i:2d}: {len(vis.split()):3d} words, font sizes {sizes}")
    if problems: raise AssertionError("\n".join(problems))

if __name__ == "__main__":
    prs = build()
    for out in OUT: out.parent.mkdir(parents=True, exist_ok=True); prs.save(str(out)); print("wrote", out)
    check(OUT[0]); print("checks passed:", len(SLIDES), "slides")
