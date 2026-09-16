# Addendum A5 — downstream exam on the intervention arms (the fault-panel validity test)

**Dated 2026-08-04, written BEFORE any arm has a downstream score. User
authorization for ~20-25 GPU-hours recorded this date. Purpose: metric
validity has only ever been measured on panels of healthy pretrained
models; real model development (and Aim 2 itself) produces degenerate
checkpoints. This exam measures every candidate's validity on a panel
that CONTAINS the failure modes.**

## Protocol (declared)

- Target: Belebele 0-shot log-likelihood accuracy via lm-evaluation-
  harness, bf16, trust_remote_code, protocol-matched to the campaign's
  healthy-model runs. Task list restricted to the 41 panel languages
  (declared list in scripts/sbatch/belebele_arms.sbatch; flores-keyed).
- Panel: all 75 trained checkpoints (25 arms x 3 seeds). Frozen bases
  are covered by the existing healthy-model runs.
- Per-arm ability statistic: mean accuracy over the 40 non-English
  tasks ("panel_acc"). Per-(arm, language) rows kept for panel analyses.
- Arm-level metric values: mexa_shared and aar_shared per arm from a
  dedicated baselines array (n=300, dip L8, z-scored shared prep — the
  grid protocol; run IDs arm_baselines_*), joining the existing
  arm_components (q_L, C_pres) and inherited q_K.
- Primary panel statistic for arm ranking, declared HERE before any
  score exists: soft-min over [q_L, q_K, q_C] (tau = 0.1), median over
  languages — the T-excluded intervention aggregate. T is reported
  BESIDE it and enters no primary claim (S1/S2 established inversion;
  this addendum extends the pre-registered T-excluded line to
  intervention mode for THIS exam).
- Validity statistic: Spearman across the 75 checkpoints between each
  metric's arm value and panel_acc, with arm-family cluster bootstrap
  (25 families, 2000 reps, seed 13) for CIs.

## Frozen predictions (two-sided; failures bind us)

- **P1 (ability).** WA-C's panel_acc is at least 5pp below the LM-only
  control (armB) and below WA-G, in >= 2 of 3 seeds. **If P1 fails, the
  record states that C_pres flagged a model whose ability was NOT
  degraded — a false-alarm finding AGAINST the panel's fault-detection
  claim — and P2/P3 are void.**
- **P2 (incumbent mis-ranking; conditional on P1).** MEXA_arm ranks each
  WA-C seed above the armB median despite lower actual ability, and
  MEXA's arm-validity Spearman is below the panel statistic's.
- **P3 (panel validity; conditional on P1).** The panel statistic's
  Spearman with panel_acc is positive with the 95% cluster CI excluding
  zero.
- **P4 (alignment null).** WA-G does not exceed armB by more than 2pp
  panel_acc (alignment training buys no downstream multilingual gain),
  consistent with the prior campaign's code-switch result.

Every prediction is scored PASS/FAIL verbatim in DECISIONS; no
redesign in response to outcomes.
