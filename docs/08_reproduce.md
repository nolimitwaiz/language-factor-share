# Reproducing

## Two levels

1. **From stored results, on a laptop.** Every table and figure in the paper is derived from files
   under `results/`, `rmfs/results/`, and `paper/iclr2027/tables/`. This needs Python 3.11 or later
   and `pip install -r requirements.txt`. No model is loaded.
2. **From scratch, on a GPU cluster.** The scripts in `slurm/` and `cluster/` regenerate the stored
   results. They were run on a SLURM cluster with one GPU per job: an RTX-class partition for models up to
   about 4B parameters and an A100 partition for larger ones; the largest model is 13B parameters.

## Environment

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # cluster only:
    pip install "torch>=2.3" "transformers>=4.46" lm-eval

`rmfs/` is also an installable package (`pip install -e rmfs`) with its own tests.

## Level 1: re-derive from stored results

| What | Command | Reads | Writes |
|---|---|---|---|
| Profile summary and claim audit | `python rmfs/scripts/audit_lfs_headline_claims.py` | `results/grid/`, `rmfs/results/runs/` | `rmfs/results/tables/lfs_*_audit.*` |
| Paper Figure 1 | `cd paper/iclr2027/figs && python make_fig1.py` | `results/grid/`, `results/dip_bootstrap/`, tables | `fig1_lfs_profiles_17.{pdf,png}` |
| Paper Figure 2 | `python make_fig_r1.py` | `results/round3/r1_content_transfer_analysis_2026-08-24/model_level_summary.csv` | `fig_r1_model_level.{pdf,png}` |
| Decomposition figure | `python make_fig15.py` | `results/budget/*_fixed_raw/budget.json` | `fig15_budget_composition.{pdf,png}` |
| Ratio-versus-component figure | `python make_fig17.py` | `results/aim2/evaluation_wordalign.json` | `fig17_ratio_hides_scale.{pdf,png}` |
| Replication and resampling scoring | `python src/analysis/score_eabc.py` | `results/flores_grid/`, `results/dip_bootstrap/`, `results/instruct_grid/` | `results/eabc_scoring/` |
| Content-transfer analysis | `python src/analysis/assemble_analyze_r1_content_transfer.py` | `results/round3/r1_content_pairs_*`, per-model outputs | `results/round3/r1_content_transfer_analysis_*` |
| Pooled-benchmark validity | `python rmfs/scripts/expanded_validity.py` | `rmfs/results/belebele_new/`, `include_native/`, `results/deflation/` | `rmfs/results/tables/expanded_validity.txt` |
| Study-guide and deck figures | `python docs/study/figs/make_*.py`; `python presentations/make_*_deck.py` | results and tables | PNGs, PPTX |
| Paper | `cd paper/iclr2027 && tectonic main.tex` (or pdfLaTeX) | | `main.pdf` |

## Level 2: regenerate the stored results

In the order the project ran them:

1. `cluster/submit_grid.sh`, `submit_grid3.sh`, `submit_big1.sh`: the primary grids (`code/cluster_grid.py`).
2. `slurm/campaign_small.sbatch`, `campaign_big.sbatch`: the 1,500-sentence dumps for 14 models (`src/campaign/dump_embeddings.py`).
3. `slurm/e_b_flores_grid.sbatch`, `e_c_instruct_grid.sbatch`: FLORES-200 replication and the instruction-tuned pairs (`code/cluster_grid_corpus.py`); `e_a_dip_bootstrap.sbatch` on the CPU partition for the resampling intervals (`src/analysis/dip_bootstrap.py`).
4. `slurm/campaign_budget.sbatch` (`src/analysis/budget_real.py`) and `slurm/round3_invariance.sbatch` (`src/analysis/round3_invariance.py`): the decomposition and the invariance suite on the dumps; `rmfs/scripts/grid_components.py` for the variance components.
5. `slurm/r1_*.sbatch`: the 19-model content-transfer array, then the gated assembly script.
6. `cluster/submit_belebele.sh`, `submit_belebele0.sh`, and the rmfs benchmark jobs: Belebele and INCLUDE through lm-evaluation-harness; `code/attribution.py` for the adjustment.
7. `slurm/aim2_*.sbatch`: the training interventions and their evaluation.

Each job writes a manifest with code hash, configuration, input hashes, seed, host, precision, and
wall time; the cluster run ledger is `ledgers/run_ledger.jsonl`.

## Tests

    pytest tests            # estimators, nulls, analysis assembly
    pytest rmfs/tests       # the rmfs package

The tests are CPU-only (76 collected under `tests/`). One module, `tests/test_round3_invariance.py`,
imports the rmfs package, so run `pip install -e rmfs` first.

## Rules the project followed

Frozen protocols are never edited, only extended by dated addenda before the affected run. Saved
dumps are read-only; new extractions get new run ids. Pooling and statistics are fp32 everywhere.
Every reported number carries its components, grid size, pooling, layer, model revision, and an
uncertainty statement, and every failed frozen prediction is reported.
