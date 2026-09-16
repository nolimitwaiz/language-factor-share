#!/usr/bin/env python3
"""Compute RMFS profile 0.2 from campaign embedding dumps (observational table).

Read-only analysis of existing dumps. Emits no 0–100 composite. When
``--selector-results`` is supplied, MEXA from that file is labeled as an
external comparison only (not an RMFS reading).

Protocol matches the independent audit job (selection [0,300), held-out [300,1500)
in 300-sentence blocks; z-score fit on selection, frozen on held-out; English
pivot + fixed 12-language panel; float32 for memory).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Allow ``python src/analysis/rmfs_profile_campaign.py`` from repo root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.analysis.rmfs_profile import PROFILE_VERSION  # noqa: E402

DEFAULT_LANGUAGES = [
    "deu", "rus", "arb", "zho-CN", "tur", "vie",
    "hin", "ben", "swa", "amh", "zul", "khm",
]
EPS = 1e-12


def standardize_from_selection(selection: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    # Keep the 7B-model path within CPU-memory limits. Source dumps are
    # float16 post-pooling; float32 is already more precise than the stored
    # values and avoids a second multi-gigabyte float64 copy.
    mean = selection.mean(axis=(0, 1), keepdims=True, dtype=np.float32)
    scale = selection.std(axis=(0, 1), keepdims=True, dtype=np.float32)
    scale = np.where(scale > EPS, scale, 1.0)
    return (evaluation.astype(np.float32, copy=False) - mean) / scale


def decomposition(x: np.ndarray) -> dict[str, float]:
    n_lang, n_concept, _ = x.shape
    grand = x.mean(axis=(0, 1))
    language_mean = x.mean(axis=1)
    concept_mean = x.mean(axis=0)
    residual = x - language_mean[:, None, :] - concept_mean[None, :, :] + grand
    ss_language = float(n_concept * np.square(language_mean - grand).sum())
    ss_concept = float(n_lang * np.square(concept_mean - grand).sum())
    ss_residual = float(np.square(residual).sum())
    systematic = ss_language + ss_concept
    total = systematic + ss_residual
    lfs = ss_language / systematic if systematic > EPS else float("nan")
    return {
        "lfs": lfs,
        "concept_dominance": 1.0 - lfs,
        "ss_language": ss_language,
        "ss_concept": ss_concept,
        "ss_residual": ss_residual,
        "language_share_total": ss_language / total if total > EPS else float("nan"),
        "concept_share_total": ss_concept / total if total > EPS else float("nan"),
        "residual_share_total": ss_residual / total if total > EPS else float("nan"),
    }


def alignment_profile(x: np.ndarray, languages: list[str]) -> dict:
    """Compute R2/R3 under the same frozen preprocessing as R1."""
    per_language = {}
    pivot = x[0]
    pivot = pivot / np.maximum(np.linalg.norm(pivot, axis=1, keepdims=True), EPS)
    for index, language in enumerate(languages, start=1):
        target = x[index]
        target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), EPS)
        similarity = target @ pivot.T
        correct = np.diag(similarity)
        wrong = np.where(
            np.eye(len(similarity), dtype=bool), -np.inf, similarity
        ).max(axis=1)
        margin = correct - wrong
        n_tail = max(1, int(np.floor(0.10 * len(margin))))
        per_language[language] = {
            "mean_margin": float(margin.mean()),
            "aar10": float(np.sort(margin)[:n_tail].mean()),
            "retrieval_success_rate": float((margin > 0).mean()),
            "n_sentences": int(len(margin)),
        }
    means = np.asarray([row["mean_margin"] for row in per_language.values()])
    tails = np.asarray([row["aar10"] for row in per_language.values()])
    n_weak = max(1, int(np.floor(0.25 * len(tails))))
    return {
        "macro_mean_margin": float(means.mean()),
        "weak_language_tail": float(np.sort(tails)[:n_weak].mean()),
        "macro_retrieval_success": float(np.mean([
            row["retrieval_success_rate"] for row in per_language.values()
        ])),
        "per_language": per_language,
    }


def selector_map(path: Path) -> tuple[dict[str, int], dict[str, dict]]:
    payload = json.loads(path.read_text())
    layer_by_tag = {}
    external = {}
    for model in payload["models"]:
        tag = model["tag"]
        selected = model["selectors"]["lfs"]
        layer_by_tag[tag] = int(selected["layer"])
        external[tag] = {
            "label": "external proxy outcome; not an RMFS reading",
            "heldout_mexa": selected["heldout_mexa"],
            "heldout_aar10_from_prior_pipeline": selected["heldout_aar10"],
        }
    return layer_by_tag, external


def dip_layers_from_dumps(dumps: Path) -> dict[str, int]:
    """Fallback layer choice: LFS-dip recorded in each dump's meta.json."""
    out: dict[str, int] = {}
    for meta_path in sorted(dumps.glob("*/meta.json")):
        tag = meta_path.parent.name
        meta = json.loads(meta_path.read_text())
        if "dip_layer" not in meta:
            raise KeyError(f"{meta_path} missing dip_layer")
        out[tag] = int(meta["dip_layer"])
    return out


def _unavailable_preservation() -> dict:
    return {
        "available": False,
        "reason": "pretrained model has no matched reference checkpoint",
    }


def profile_one(
    dumps: Path,
    tag: str,
    layer: int,
    languages: list[str],
    *,
    selection_start: int,
    selection_end: int,
    evaluation_start: int,
    evaluation_end: int,
    evaluation_block_size: int,
    external: dict | None,
) -> dict:
    layer_path = dumps / tag / f"layer{layer:03d}.npz"
    if not layer_path.exists():
        raise FileNotFoundError(f"missing dump for {tag} layer {layer}")

    with np.load(layer_path) as archive:
        all_languages = [str(item) for item in archive["langs"]]
        requested = ["eng", *languages]
        missing = sorted(set(requested) - set(all_languages))
        if missing:
            raise ValueError(f"{tag} lacks languages: {missing}")
        indices = [all_languages.index(language) for language in requested]
        full = archive["X"]
        selection = full[
            indices, selection_start:selection_end
        ].astype(np.float32)
        evaluation = full[
            indices, evaluation_start:evaluation_end
        ].astype(np.float32)
        del full

    z_selection = standardize_from_selection(selection, selection)
    selection_decomp = decomposition(z_selection)
    selection_margins = alignment_profile(z_selection, languages)

    if (evaluation_end - evaluation_start) % evaluation_block_size:
        raise ValueError("evaluation range must divide into complete blocks")
    block_decomp = []
    block_alignment = []
    for offset in range(0, evaluation.shape[1], evaluation_block_size):
        block = evaluation[:, offset:offset + evaluation_block_size]
        z_block = standardize_from_selection(selection, block)
        block_decomp.append(decomposition(z_block))
        block_alignment.append(alignment_profile(z_block, languages))
    evaluation_decomp = {
        key: float(np.mean([row[key] for row in block_decomp]))
        for key in block_decomp[0]
    }
    evaluation_margins = {
        "macro_mean_margin": float(np.mean([
            row["macro_mean_margin"] for row in block_alignment
        ])),
        "weak_language_tail": float(np.mean([
            row["weak_language_tail"] for row in block_alignment
        ])),
        "macro_retrieval_success": float(np.mean([
            row["macro_retrieval_success"] for row in block_alignment
        ])),
        "blocks": block_alignment,
    }
    supporting = {
        "selection_decomposition": selection_decomp,
        "heldout_decomposition": evaluation_decomp,
        "heldout_decomposition_blocks": block_decomp,
        "selection_alignment": selection_margins,
        "heldout_alignment": evaluation_margins,
    }
    if external is not None:
        supporting["prior_external_comparison"] = external
    return {
        "model_tag": tag,
        "layer": layer,
        "unit": "model/layer on fixed English-plus-12-language NTREX grid",
        "selection_profile": {
            "R1_concept_dominance": selection_decomp["concept_dominance"],
            "R2_mean_alignment_margin": selection_margins["macro_mean_margin"],
            "R3_weak_language_tail": selection_margins["weak_language_tail"],
            "R4_preservation": _unavailable_preservation(),
        },
        "heldout_profile": {
            "R1_concept_dominance": evaluation_decomp["concept_dominance"],
            "R2_mean_alignment_margin": evaluation_margins["macro_mean_margin"],
            "R3_weak_language_tail": evaluation_margins["weak_language_tail"],
            "R4_preservation": _unavailable_preservation(),
        },
        "supporting": supporting,
        "composite_score": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="results/dumps")
    parser.add_argument(
        "--selector-results",
        default=None,
        help="Aim 3 layer-selector JSON (lfs layer + optional external MEXA)",
    )
    parser.add_argument(
        "--layer-source",
        choices=("selector", "dip_meta"),
        default="selector",
        help="selector: --selector-results required; dip_meta: dump meta.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-start", type=int, default=0)
    parser.add_argument("--selection-end", type=int, default=300)
    parser.add_argument("--evaluation-start", type=int, default=300)
    parser.add_argument("--evaluation-end", type=int, default=1500)
    parser.add_argument("--evaluation-block-size", type=int, default=300)
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    args = parser.parse_args()

    dumps = Path(args.dumps)
    external: dict[str, dict] = {}
    if args.layer_source == "selector":
        if not args.selector_results:
            raise SystemExit("--selector-results is required when --layer-source=selector")
        layer_by_tag, external = selector_map(Path(args.selector_results))
    else:
        layer_by_tag = dip_layers_from_dumps(dumps)

    profiles = []
    for tag, layer in sorted(layer_by_tag.items()):
        profiles.append(
            profile_one(
                dumps,
                tag,
                layer,
                args.languages,
                selection_start=args.selection_start,
                selection_end=args.selection_end,
                evaluation_start=args.evaluation_start,
                evaluation_end=args.evaluation_end,
                evaluation_block_size=args.evaluation_block_size,
                external=external.get(tag),
            )
        )
        print(tag, layer, profiles[-1]["heldout_profile"], flush=True)

    output = {
        "version": PROFILE_VERSION,
        "created_from": os.path.abspath(args.dumps),
        "language_panel": ["eng", *args.languages],
        "selection_sentences": [args.selection_start, args.selection_end],
        "heldout_sentences": [args.evaluation_start, args.evaluation_end],
        "heldout_block_size": args.evaluation_block_size,
        "standardization": "fit on selection split, frozen on held-out split",
        "layer_source": args.layer_source,
        "profiles": profiles,
        "warning": "No composite score. MEXA remains an external proxy comparison.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
