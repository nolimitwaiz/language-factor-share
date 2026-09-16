# PRE-REGISTRATION: Aim 3 factual mRAG (staged)
# STATUS: DRAFT — Stage 1 protocol ACTIVE for human evidence review.
# Stages 2–4 are EMBARGOED until the Stage 1 gate passes and this document
# is frozen with approved example IDs + language panel confirmation.
#
# Author: Waiz Khan
# Drafted: 2026-08-12
# Sources: docs/NEXT_EXPERIMENTS.md; Codex Aim 3 Stage 0 audit + review protocol

## 0. Scope and non-goals

**In scope:** staged factual claim verification / evidence-conditioned
prediction on X-FACT (engineering pilot → confirmatory scale).

**Out of scope until Stage 1 gate:** GPU jobs that treat raw X-FACT search
snippets as gold evidence; end-to-end RAG scores; treating NTREX geometry as
the primary Aim 3 claim; calling MEXA an Aim 3 outcome metric; calling the
NTREX layer-selector proxy a RAG result.

## 1. Stage 0 (done — reference)

- Official X-FACT release audited; leak URLs/claims documented.
- Provisional languages: **Arabic, Hindi, Indonesian, Turkish**.
  Engineering lock 2026-08-17 keeps this panel for the engineering packet
  (`docs/ENGINEERING_DECISIONS_2026-08-17.md`). Murray confirmation is for
  later confirmatory scale. Hindi has no `true` in the as-built 100-packet.
- Review packet: 100 examples (25×4), relevance fields blank.
- In-repo copies: `docs/aim3/XFACT_STAGE0_AUDIT.md`,
  `docs/aim3/EVIDENCE_REVIEW_PROTOCOL.md`,
  `data/aim3/xfact_dev_evidence_review_100.jsonl`.

## 2. Stage 1 — evidence review (ACTIVE)

### 2.1 Unit and labels

One X-FACT example + up to 5 candidate snippets. Per passage:

- `relevance` ∈ {direct, partial, irrelevant, unclear}
- `contains_verdict_leakage` (bool)
- `usable_as_positive_passage` (bool)
- `review_notes` (short)

Then `example_approved` = true iff ≥1 usable positive **and** ≥1 credible
irrelevant / hard-negative.

### 2.2 Exclusions

1. No snippet supports or refutes the claim.
2. Only useful snippet reveals the fact-check verdict.
3. Claim truncated / uninterpretable.
4. Label cannot map consistently to the three-way task.
5. Evidence needs missing media / inaccessible page.
6. Positive and negative cannot be distinguished confidently.

### 2.3 Gate (blocks Stages 2–4)

- **≥15 approved examples per language.**
- Double-review subset for agreement before full use.
- Hindi note: no `true` in the as-built 100-packet — no label-balanced
  cross-language accuracy from this packet alone; report as limitation or redraw.

### 2.4 Outputs before freeze

- Filled review JSONL + disagreement log
- Frozen approved-ID list + fact-level split hashes (translations stay together)
- One-page Stage 1 report: approval rates, leakage rates, agreement

## 3. Stages 2–4 (EMBARGOED — design text only)

Do not run until Stage 1 gate + freeze signature below. Full design lives in
`docs/NEXT_EXPERIMENTS.md` §4. Summary:

| Stage | RQ (short) | Exit |
|---|---|---|
| 2 Gold-evidence gen | Generator uses correct cross-lang evidence? | Gold > none and gold > irrelevant; compliance OK |
| 3 Dense retrieval | LFS-selected layer vs final on labeled Recall@5? | Paired bootstrap CI for (LFS − final) |
| 4 E2E RAG | Separate generator / retrieval / ranking losses | Task + faithfulness + compliance |

**Family validation (when Stages 2–4 run):** R1→layer selection; R2→macro
retrieval / evidence-use; R3→worst-lang; R4→safety on trained ckpts only.
No averaging. MEXA not an Aim 3 outcome.

**Compute:** stop-and-ask before any job estimated >~1 GPU-hour.

## 4. Shared protocol (for freeze)

- Task: factual claim verification / QA-style label with evidence.
- Split: fact-level; never put translations of the same fact across splits.
- Panel (provisional): ar, hi, id, tr — confirm with Murray before confirmatory scale.
- Engineering scale: 100 dev + 100 test facts; 1 small decoder + 1 instruct generator.
- NTREX: engineering only; never primary Aim 3 claim.

## 5. Freeze signature

```
STATUS: DRAFT (Stage 1 active; Stages 2–4 embargoed)
Stage 1 completed date: ________
Approved counts (ar/hi/id/tr): ________
Frozen date (authorizes Stages 2–4): ________
Frozen by: Waiz Khan
Git commit / tag: ________
Approved ID list path: ________
Advisor language-panel OK: ________
```
