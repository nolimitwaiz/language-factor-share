# Engineering decisions (locked 2026-08-17)

**Status:** locked for this quarter’s engineering work. Not a frozen
preregistration signature and not advisor confirmation of confirmatory
scale.  
**Author:** Waiz Khan  
**Source session:** plan-only lock after WP A0/A1; **no training authorized**.  
**HTML recap:** `reports/session_2026-08-17_plan_lock.html`

These four items were previously listed as open advisor questions in
`docs/NEXT_EXPERIMENTS.md` §8. They are now decided for engineering use.
Murray/Koehn input remains welcome for later confirmatory scale and paper
naming, but it is not a gate for B1 or Aim 1 packaging.

---

## 1. Name

**Lock:** keep **RMFS** only as the *campaign* that stress-tested a scalar.
Do not use RMFS as the name of Profile 0.2 in new writing. Call the compact
vector a **diagnostic profile** (R1–R4).

Matches the LFS-led submission (`docs/SUBMISSION_DECISION_LFS_VS_RMFS.md`)
and avoids implying a validated 0–100 score.

## 2. Aim 3 engineering panel

**Lock:** **X-FACT + {ar, hi, id, tr}**.

Stage 0 already showed enough strict-valid development rows, three scripts,
and mixed resource. Murray confirmation is for later scale-up, not for the
100-example engineering packet.

Hindi has no `true` in the as-built 100-packet. Treat that as a stated
limitation, not a reason to redraw before Stage 1.

Packet: `data/aim3/xfact_dev_evidence_review_100.jsonl`  
Protocol: `docs/aim3/EVIDENCE_REVIEW_PROTOCOL.md`

## 3. Scope this quarter

**Lock:** **Aim 1 packaging first, then Aim 3 staging. No new Aim 2
training.**

Completed Aim 2 negatives already rule out naive LFS minimization, word MSE,
and CVP-as-safety. A task-anchored contrastive + KL screen is a *later*
study with its own prereg, power analysis, and GPU-hour ask. Draft that
protocol after Wednesday only if a next-loss story is requested. Do not run
it now.

## 4. Canonical LFS estimator

**Lock:** **direct sums-of-squares** is the paper estimator:

`LFS = SS_language / (SS_language + SS_concept)` after coordinate z-score.

REML LFS-VC is a robustness check. On the same-support comparison they
ranked almost identically; they remain mathematically distinct. Profile R1
stays `1 −` the direct-SS ratio.

Implementation: `code/pilot_metrics.py` and
`src/analysis/rmfs_profile.factor_decomposition`.

---

## What this does and does not authorize

| Authorized now | Not authorized |
|---|---|
| Aim 1 manuscript estimator language (direct SS + components) | Any new training / new loss |
| Human X-FACT evidence review (B1) | RAG GPU (gold gen, retrieval, e2e) |
| Optional freeze of `PREREG_FAMILY_V0` before A2/A3 scoring of *existing* artifacts | Aim 2 co-design GPU |
| Costing and writing Stage 2–4 protocols (still embargoed) | Walk-forward extracts blocked on HF token / licenses |

**B1 remains human.** The panel and “no GPU until the gate” are decided.
Relevance labels cannot be invented.

## Order after Wednesday (2026-08-19)

1. Lock Aim 1 paper estimator and table footnotes (direct SS + components).
2. Finish B1. No GPU until ≥15 approved examples per language.
3. Freeze Aim 3 Stage 2 protocol + GPU-hour estimate; ask before submit.
4. Only then: gold-evidence generation, then retrieval, then e2e.
5. Aim 2 next-loss screen is a separate freeze, only if wanted this quarter.
