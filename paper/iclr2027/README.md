# ICLR 2027 manuscript (anonymous)

Deadlines (verified on iclr.cc 2026-09-04): abstract **2026-09-18 AOE**,
full paper **2026-09-25 AOE**. Main text at most 9 pages; references,
appendices, AI-use / ethics / reproducibility statements do not count.

## Build

```
cd paper/iclr2027
tectonic main.tex          # or: pdflatex + bibtex + pdflatex x2
```

Official style files are in `style/` (downloaded 2026-09-04) and copied to the
tree root. Do not edit them.

## Files

- `main.tex` — the paper. `\todo{}` marks are red in the PDF; all must be gone
  before submission. `\iclrfinalcopy` stays commented out for submission.
- `references.bib` — entries marked `VERIFY` need a primary-source check.
- `claims_ledger.csv` — every number in the paper mapped to artifact, code,
  job ID, and verification status. Rows marked "NOT re-verified today" were
  taken from the technical report or ledgers and should be re-read from the
  artifact before the camera-ready.
- `figs/` — placeholders copied from `../figs`. Figure 1 must be regenerated
  from audited artifacts (Mistral's early minimum is missing in the July
  version). The R1 model-level scatter still needs to be produced from
  `results/round3/r1_content_transfer_analysis_2026-08-24/model_level_summary.csv`.

## Anonymity checklist

No author names, institution, cluster name, grant or proposal references,
advisor names, or self-identifying URLs in `main.tex`, figures, or
supplementary files. Cite own prior work in the third person only.
