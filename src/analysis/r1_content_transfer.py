#!/usr/bin/env python3
"""Batched, revision-locked Aim 1 R1 cross-lingual content scoring."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np


EXPECTED_PAIR_SHA256 = "986d4f3a8730b7e8087ebf0d558de72e8735aba3b4fb5493a3f1e04b42bc1112"
EXPECTED_DATA_MANIFEST_SHA256 = "7c62075cdda2cecd4dfb3c3e06664c6fc9eae17e487eb4ef9c01be5a41d05fcd"
EXPECTED_MODEL_LOCK_SHA256 = "dcb3c1fbaffa8cca70a4090b3b62d269a5daddc43779e62a9e2cdede8679930c"
SMOKE_LANGUAGES = ["eng", "deu", "khm"]
MAX_SEQUENCE_TOKENS = 2048


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, observed {observed}")
    return observed


def load_model_lock(path: Path, tag: str) -> dict[str, str | int]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    matches = [row for row in rows if row["tag"] == tag]
    if len(rows) != 19 or len(matches) != 1:
        raise ValueError(f"expected one {tag} row in a 19-model lock")
    row = dict(matches[0])
    row["batch_size"] = int(row["batch_size"])
    return row


def read_pairs(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100:
        raise ValueError(f"expected 100 frozen pairs, found {len(rows)}")
    for row in rows:
        for column in ("pair_id", "matched_context_index", "target_index", "mismatched_context_index"):
            row[column] = int(row[column])
    return rows[:limit] if limit is not None else rows


def load_texts(ntrex: Path, languages: list[str], source_hashes: dict[str, str]) -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {}
    for language in languages:
        path = ntrex / (
            "newstest2019-src.eng.txt" if language == "eng"
            else f"newstest2019-ref.{language}.txt"
        )
        if sha256(path) != source_hashes[language]:
            raise ValueError(f"source hash mismatch for {language}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1997:
            raise ValueError(f"{language}: expected 1997 rows, found {len(lines)}")
        texts[language] = lines
    return texts


def build_specs(
    pairs: list[dict[str, Any]], texts: dict[str, list[str]], languages: list[str]
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    english = texts["eng"]
    for pair in pairs:
        target = english[pair["target_index"]]
        specs.append({
            "pair_id": pair["pair_id"], "language": "eng", "condition": "none",
            "context": None, "target": target,
        })
        for language in languages:
            specs.append({
                "pair_id": pair["pair_id"], "language": language, "condition": "matched",
                "context": texts[language][pair["matched_context_index"]], "target": target,
            })
            specs.append({
                "pair_id": pair["pair_id"], "language": language, "condition": "mismatched",
                "context": texts[language][pair["mismatched_context_index"]], "target": target,
            })
    return specs


def encode_spec(tokenizer: Any, spec: dict[str, Any]) -> dict[str, Any]:
    target_ids = tokenizer.encode(" " + spec["target"], add_special_tokens=False)
    if not target_ids:
        target_ids = tokenizer.encode(spec["target"], add_special_tokens=False)
    if not target_ids:
        raise ValueError("target tokenized to zero tokens")
    if spec["context"] is None:
        anchor = tokenizer.bos_token_id
        if anchor is None:
            anchor = tokenizer.eos_token_id
        if anchor is None:
            raise ValueError("tokenizer has neither BOS nor EOS for no-context scoring")
        prefix_ids = [int(anchor)]
    else:
        prefix_ids = tokenizer.encode(spec["context"] + "\n", add_special_tokens=True)
        if not prefix_ids:
            raise ValueError("context tokenized to zero tokens")
    input_ids = [*prefix_ids, *target_ids]
    if len(input_ids) > MAX_SEQUENCE_TOKENS:
        raise ValueError(
            f"sequence exceeds frozen {MAX_SEQUENCE_TOKENS}-token ceiling: "
            f"{len(input_ids)}"
        )
    return {
        **{key: spec[key] for key in ("pair_id", "language", "condition")},
        "input_ids": input_ids,
        "target_start": len(prefix_ids),
        "target_tokens": len(target_ids),
        "context_tokens": len(prefix_ids) if spec["context"] is not None else 0,
        "total_tokens": len(input_ids),
    }


def target_mean_nll(logits: Any, input_ids: Any, target_start: int) -> float:
    import torch.nn.functional as functional

    stop = int(input_ids.shape[0])
    prediction = logits[target_start - 1 : stop - 1].float()
    labels = input_ids[target_start:stop]
    if prediction.shape[0] != labels.shape[0] or labels.numel() == 0:
        raise ValueError("target/logit alignment failure")
    return float(functional.cross_entropy(prediction, labels, reduction="mean").item())


def score_batches(model: Any, tokenizer: Any, encoded: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.bos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no usable padding token")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long, device="cuda")
        mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        for index, row in enumerate(batch):
            values = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            ids[index, : len(values)] = values
            mask[index, : len(values)] = 1
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, use_cache=False)
        for index, row in enumerate(batch):
            length = len(row["input_ids"])
            nll = target_mean_nll(output.logits[index, :length], ids[index, :length], row["target_start"])
            if not math.isfinite(nll):
                raise FloatingPointError("non-finite target NLL")
            rows.append({
                **{key: row[key] for key in (
                    "pair_id", "language", "condition", "target_tokens",
                    "context_tokens", "total_tokens",
                )},
                "mean_target_nll": nll,
            })
        del output, ids, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[progress] {min(start + batch_size, len(encoded))}/{len(encoded)}", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]], languages: list[str]) -> dict[str, Any]:
    none = [row["mean_target_nll"] for row in rows if row["condition"] == "none"]
    base = float(np.mean(none))
    per_language: dict[str, dict[str, float | None]] = {}
    for language in languages:
        matched = [row["mean_target_nll"] for row in rows if row["language"] == language and row["condition"] == "matched"]
        mismatched = [row["mean_target_nll"] for row in rows if row["language"] == language and row["condition"] == "mismatched"]
        if len(matched) != len(none) or len(mismatched) != len(none):
            raise ValueError(f"incomplete scored rows for {language}")
        content = float(np.mean(np.asarray(mismatched) - np.asarray(matched)))
        per_language[language] = {
            "delta_matched": float(base - np.mean(matched)),
            "delta_mismatched": float(base - np.mean(mismatched)),
            "content": content,
            "R_content": None,
        }
    english_content = per_language["eng"]["content"]
    if english_content is not None and english_content > 0:
        for language in languages:
            per_language[language]["R_content"] = float(
                per_language[language]["content"] / english_content
            )
    return {"base_nll": base, "english_content_positive": bool(english_content > 0), "per_language": per_language}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--ntrex", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    verify_hash(args.model_lock, EXPECTED_MODEL_LOCK_SHA256, "model lock")
    verify_hash(args.pairs, EXPECTED_PAIR_SHA256, "pair file")
    verify_hash(args.data_manifest, EXPECTED_DATA_MANIFEST_SHA256, "data manifest")
    model_row = load_model_lock(args.model_lock, args.tag)
    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    languages = SMOKE_LANGUAGES if args.smoke else data_manifest["languages"]
    pair_limit = 2 if args.smoke else None
    pairs = read_pairs(args.pairs, pair_limit)
    texts = load_texts(args.ntrex, languages, data_manifest["source_sha256"])
    specs = build_specs(pairs, texts, languages)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("R1 scoring requires a CLSP CUDA node")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_row["hf_id"], revision=model_row["revision"], local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_row["hf_id"], revision=model_row["revision"], local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    encoded = [encode_spec(tokenizer, spec) for spec in specs]
    rows = score_batches(model, tokenizer, encoded, int(model_row["batch_size"]))
    summary = summarize(rows, languages)

    output_dir = args.output_root / ("smoke" if args.smoke else "parts") / args.tag
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "per_example.csv"
    summary_path = output_dir / "summary.json"
    write_csv(rows_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": f"aim1_r1_{'smoke' if args.smoke else 'full'}_{args.tag}",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tag": args.tag,
        "hf_id": model_row["hf_id"],
        "revision": model_row["revision"],
        "model_family": model_row["model_family"],
        "batch_size": int(model_row["batch_size"]),
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "smoke": args.smoke,
        "n_pairs": len(pairs),
        "languages": languages,
        "n_rows": len(rows),
        "expected_rows": len(pairs) * (1 + 2 * len(languages)),
        "all_nll_finite": bool(all(math.isfinite(row["mean_target_nll"]) for row in rows)),
        "model_lock_sha256": EXPECTED_MODEL_LOCK_SHA256,
        "pairs_sha256": EXPECTED_PAIR_SHA256,
        "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "per_example_sha256": sha256(rows_path),
        "summary_sha256": sha256(summary_path),
        "cuda_device": torch.cuda.get_device_name(0),
        "wall_seconds": round(time.time() - started, 3),
        "outcomes_printed_to_log": False,
    }
    if manifest["n_rows"] != manifest["expected_rows"] or not manifest["all_nll_finite"]:
        raise ValueError("R1 completeness/finite-value contract failed")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "run_id": manifest["run_id"],
        "n_rows": manifest["n_rows"],
        "all_nll_finite": manifest["all_nll_finite"],
        "revision": manifest["revision"],
        "wall_seconds": manifest["wall_seconds"],
        "outcomes_printed_to_log": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
