# Training interventions

These studies ask what happens when a measure is used as a training signal. Every one fine-tunes
`Qwen/Qwen3-0.6B-Base` for 400 steps at learning rate 1e-5 on NTREX sentences 500 to 1,899 in
English, German, Hindi, and Swahili, with batches built as sentence x language grids so that every
loss term and every monitor (LFS, content variance, effective rank) can be computed on the same
forward pass. Three seeds (0, 1, 2) per condition. Evaluation after training reads the layer-8
representation grid (`src/aim2/evaluate.py`: LFS and its components, MEXA, AaR, CVP, effective rank,
norm), the context-use behavior (`src/aim2/context_use.py`), and Belebele accuracy on the trained
checkpoints (`rmfs/results/belebele_arms/`). Plans and scorecards: `docs/AIM2_STUDY_PLAN.md`,
`docs/AIM2_STUDY_SCORECARD.md`, `docs/WORDLEVEL_STAGE1_SCORECARD.md`, `docs/CODESWITCH_SCORECARD.md`.

## Sentence-level alignment (proposal section 3.2.2, first form)

`src/aim2/train.py`, job `slurm/aim2_pilot.sbatch` and `slurm/aim2_sweep.sbatch`.

| Condition | Objective |
|---|---|
| A | frozen checkpoint, no training |
| B | language-model loss only (domain-adaptation control) |
| C | language-model loss + LFS as a loss |
| D | language-model loss + contrastive alignment (InfoNCE across translations) |
| E | D + a content-variance preservation hinge, CVP, with floor 0.90 |
| F | E + LFS |
| G | E with a differentiable tail-margin term (drives AaR up directly) |
| E-cvp10, E-cvp100, E-cvp1000 | condition E with the hinge weight raised to 10, 100, 1,000 |

CVP here is the within-language variance across sentences, summed over coordinates and averaged
over languages, relative to the frozen reference (`concept_variance` in `src/aim2/train.py`).

**Outcome.** Across 18 condition-seed points from six conditions, LFS and tail alignment moved
together (Pearson +0.92): the objectives that drove LFS down drove AaR down with it, the opposite of
the association observed across pretrained models. Training on LFS itself doubled the dip in 400
steps while Belebele fell 2.9 points against the control's 1.7 (`code/aim2_pilot.py`,
`results/grid/aim2-*`). The CVP hinge saturated at every weight tested and did not prevent the
contraction under this setup. `results/aim2/evaluation.json`, `results/aim2/evaluation_sweep.json`.

## Word-level alignment (proposal section 3.2.2, second form)

Protocol: `prereg/PREREG_WORDALIGN.md`. `src/aim2/train_wordalign.py`, job `slurm/aim2_wordalign.sbatch`.
Word links are agreed one-to-one alignments between a sentence and its translation.

| Condition | Objective |
|---|---|
| WA-A | frozen checkpoint |
| WA-B | language-model loss only |
| WA-C | + squared difference between linked word representations (the objective from the literature, Cao et al. 2020) |
| WA-D | WA-C + a learned language vector |
| WA-E | + segment-level squared difference instead of word-level |
| WA-F | WA-D + the CVP hinge |
| WA-G | WA-D with cosine distance instead of squared difference (scale-invariant, no h = 0 minimum) |

**Outcome.** The squared-difference objective has a degenerate minimum at h = 0 and reached toward
it: representation norm fell from 451 to 42 and raw content variance from 13,791 to 699 (twenty-fold),
while LFS moved only from 0.445 to 0.493 and MEXA rose by +0.09 over the control. Belebele stayed
at 0.395 against 0.390 for the control, so this is a representation change, not demonstrated harm.
The cosine variant did not shrink and gained nothing. `results/aim2/evaluation_wordalign.json`.

## Code-switched training (proposal section 3.2.1)

Protocol: `prereg/PREREG_CODESWITCH.md`. `src/aim2/train_codeswitch.py`, `src/aim2/codeswitch.py`,
job `slurm/aim2_codeswitch.sbatch`. A replay mixture held constant across conditions; CS-B is the
unswitched control, CS-W10, CS-W25, CS-W50 switch 10, 25, 50 percent of words, CS-P25 switches 25
percent of phrases. **Outcome.** No detectable benefit on the frozen criteria and a measurable cost
outside the replay languages (`docs/CODESWITCH_SCORECARD.md`, `results/aim2/evaluation_codeswitch.json`).

## What these studies establish, and what they do not

They show what the measures do under these objectives on one 0.6B model. They do not identify the
causes of the cross-model pattern, and they do not prove that every LFS-based objective must fail.
The paper's conclusion is narrower: LFS is a measurement to report with its raw components, not a
standalone score or loss.
