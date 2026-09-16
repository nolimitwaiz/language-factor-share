# PRE-REGISTRATION: Metric family + RMFS Profile 0.2 (v0)
# STATUS: DRAFT — not frozen. Design text copied from docs/NEXT_EXPERIMENTS.md
# (2026-08-12). Freeze only by dated Waiz signature / git tag before inspecting
# confirmatory A2/A3 outcome tables. A0/A1 (merge code + reproduce prior audit
# table) may run under this draft; A2–A5 scoring against P-F* requires freeze.
#
# Author: Waiz Khan
# Drafted: 2026-08-12

## 0. Scope

Package Aim 1 representation diagnostics and Aim 2 training monitors as an
explicit **purpose-specific metric family**, with **RMFS Profile 0.2** as the
compact observational / intervention profile over that family.

**Not in scope for this prereg:** new Aim 1 grids; new minimize-LFS / word-MSE
alignment arms; publishing a 0–100 composite “RMFS score”; calling MEXA an
original reading.

### Profile 0.2 readings

| ID | Reading | Definition (summary) | Availability |
|---|---|---|---|
| R1 | Concept dominance | `1 − LFS`, LFS = committed SS ratio after z-score | Always (grid) |
| R2 | Mean parallel margin | Macro mean over langs of mean_c (diag − max hard neg) | Always |
| R3 | Weakest-language tail | Mean of lowest β=0.25 of per-lang AaR@10 | Always |
| R4 | Preservation gate | Pass iff `CVP_within ≥ 0.90` vs frozen reference | Intervention only; else `unavailable` |

**Hard rules**

1. No soft-min / percentile average / 0–100 composite of R1–R3.
2. MEXA is an **external baseline column** only.
3. Always publish the reading vector plus supporting component tables
   (`ss_language`, `ss_concept`, margins, etc.).
4. Never claim lower LFS alone means better capability.
5. Canonical LFS estimator for claims: committed direct SS ratio
   (`src/analysis/rmfs_profile.factor_decomposition` / `code/pilot_metrics.py`).
   Engineering lock 2026-08-17 (`docs/ENGINEERING_DECISIONS_2026-08-17.md`):
   direct SS is the paper estimator; REML LFS-VC is robustness only. This
   document remains **DRAFT** until the freeze signature below; the lock
   does not by itself authorize A2/A3 confirmatory scoring.

### Observational campaign table (A1)

- Unit: one model at its LFS-dip layer (from Aim 3 selector JSON or dump
  `meta.json`), English + fixed 12-language NTREX panel.
- Selection sentences `[0, 300)`; held-out `[300, 1500)` in blocks of 300.
- Z-score: fit on selection, frozen on held-out (float32 path).
- R4: `unavailable` for pretrained observational rows.

## 1. Predictions (freeze before A2/A3 scoring)

- **P-F1:** R1 predicts held-out content transfer after deflation better than
  chance (bar: residual Spearman ≥ 0.30, language-cluster CI excludes 0) on the
  3.1.3 model set / extended harness.
- **P-F2:** R2 and shared-prep MEXA are positively associated; R2 is *not*
  claimed superior a priori.
- **P-F3:** On the Aim 2 fault panel, R3 or AaR correlates with `panel_acc`
  more than R1 (direction: higher R3 → higher acc).
- **P-F4:** R4 fails on WA-C-class collapse and passes on WA-G; R4 does **not**
  imply better Belebele.
- **P-F5:** No scalar average of R1–R3 beats the best single reading on both
  content-transfer and fault-panel exams simultaneously (anti-composite check).

## 2. Experiments under this prereg

| ID | Allowed under DRAFT? | Notes |
|---|---|---|
| A0 Port Profile 0.2 + tests | Yes | Code/tests only |
| A1 Reproduce 14-model table | Yes | Match prior audit within tol or ledger delta |
| A2 Family vs Belebele / transfer | **Freeze first** | Deflated Spearman per reading |
| A3 Family on Aim 2 fault panel | **Freeze first** | Correlate with panel_acc |
| A4 Estimator reconciliation note | Yes (methods) | Choose/confirm canonical estimator |
| A5 CS-2000 eval (optional) | Ask | Few GPU-h; not required for family freeze |
| A6 Walk-forward (optional) | Blocked | HF token / licenses |

## 3. Reporting language

- Original readings (R1–R4 / LFS components / AaR) vs external baselines (MEXA)
  vs task outcomes — never mixed as one “score.”
- Manifest: git hash, dump paths, selector path, host, job ID.

## 4. Freeze signature

```
STATUS: DRAFT
Frozen date: ________
Frozen by: Waiz Khan
Git commit / tag: ________
Notes: ________
```
