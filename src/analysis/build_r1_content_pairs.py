#!/usr/bin/env python3
"""Freeze document-distinct NTREX pairs for Aim 1 R1 content transfer."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
from scipy.optimize import linear_sum_assignment


LANGUAGES = [
    "eng", "deu", "fra", "spa", "por", "ita", "nld", "swe", "pol",
    "ces", "ron", "rus", "ukr", "ell", "tur", "vie", "ind", "zho-CN",
    "jpn", "kor", "arb", "heb", "fas", "hin", "ben", "tam", "tel",
    "mar", "urd", "tha", "khm", "mya", "swa", "amh", "som", "hau",
    "zul", "kat", "hye", "aze-Latn", "uzb",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_one_pair_per_document(
    english: list[str], docs: list[str], n_pairs: int
) -> list[int]:
    if len(english) != len(docs):
        raise ValueError("English and document-ID lengths differ")
    selected: list[int] = []
    used_docs: set[str] = set()
    for index in range(len(english) - 1):
        doc = docs[index]
        eligible = (
            doc == docs[index + 1]
            and len(english[index].split()) >= 5
            and 8 <= len(english[index + 1].split()) <= 45
        )
        if eligible and doc not in used_docs:
            selected.append(index)
            used_docs.add(doc)
            if len(selected) == n_pairs:
                break
    if len(selected) != n_pairs:
        raise ValueError(f"expected {n_pairs} distinct-document pairs, found {len(selected)}")
    return selected


def minimum_cost_derangement(indices: list[int], english: list[str]) -> list[int]:
    n = len(indices)
    lengths = np.asarray([len(english[index].split()) for index in indices], dtype=np.int64)
    positions = np.arange(n, dtype=np.int64)
    cost = (
        np.abs(lengths[:, None] - lengths[None, :]) * 1_000_000
        + np.abs(positions[:, None] - positions[None, :]) * 1_000
        + positions[None, :]
    ).astype(np.float64)
    np.fill_diagonal(cost, 1e15)
    row, column = linear_sum_assignment(cost)
    if not np.array_equal(row, positions) or np.any(column == positions):
        raise ValueError("failed to construct deterministic derangement")
    mismatched = [indices[int(candidate)] for candidate in column]
    if len(set(mismatched)) != n:
        raise ValueError("mismatched contexts are not one-to-one")
    return mismatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntrex", type=Path, required=True)
    parser.add_argument("--document-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=100)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {args.output_dir}")
    english_path = args.ntrex / "newstest2019-src.eng.txt"
    english = english_path.read_text(encoding="utf-8").splitlines()
    docs = [line.split("\t")[0] for line in args.document_ids.read_text(encoding="utf-8").splitlines()]
    if len(english) != 1997 or len(docs) != 1997:
        raise ValueError("expected 1,997 NTREX rows")

    source_paths: dict[str, Path] = {"eng": english_path}
    for language in LANGUAGES[1:]:
        source_paths[language] = args.ntrex / f"newstest2019-ref.{language}.txt"
    for language, path in source_paths.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(english):
            raise ValueError(f"{language}: expected {len(english)} rows, found {len(lines)}")

    selected = select_one_pair_per_document(english, docs, args.n_pairs)
    mismatched = minimum_cost_derangement(selected, english)
    rows = []
    for pair_id, (matched, mismatch) in enumerate(zip(selected, mismatched)):
        if docs[matched] == docs[mismatch]:
            raise ValueError("mismatched context shares target document")
        rows.append({
            "pair_id": pair_id,
            "matched_context_index": matched,
            "target_index": matched + 1,
            "mismatched_context_index": mismatch,
            "matched_document_id": docs[matched],
            "mismatched_document_id": docs[mismatch],
            "matched_english_context_words": len(english[matched].split()),
            "mismatched_english_context_words": len(english[mismatch].split()),
            "target_english_words": len(english[matched + 1].split()),
        })

    args.output_dir.mkdir(parents=True)
    pairs_path = args.output_dir / "pairs.csv"
    with pairs_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "run_id": "aim1_r1_content_pairs_100_documents",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": socket.gethostname(),
        "n_pairs": len(rows),
        "n_unique_matched_documents": len({row["matched_document_id"] for row in rows}),
        "n_unique_mismatched_documents": len({row["mismatched_document_id"] for row in rows}),
        "languages": LANGUAGES,
        "selection": "first eligible pair per document; first 100 documents",
        "derangement": "Hungarian minimum English word-length difference, then index distance and candidate index",
        "document_ids_sha256": sha256(args.document_ids),
        "source_sha256": {language: sha256(path) for language, path in source_paths.items()},
        "pairs_sha256": sha256(pairs_path),
        "maximum_english_context_word_difference": max(
            abs(row["matched_english_context_words"] - row["mismatched_english_context_words"])
            for row in rows
        ),
        "mean_english_context_word_difference": float(np.mean([
            abs(row["matched_english_context_words"] - row["mismatched_english_context_words"])
            for row in rows
        ])),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_pairs": manifest["n_pairs"],
        "n_unique_matched_documents": manifest["n_unique_matched_documents"],
        "pairs_sha256": manifest["pairs_sha256"],
        "maximum_english_context_word_difference": manifest["maximum_english_context_word_difference"],
        "mean_english_context_word_difference": manifest["mean_english_context_word_difference"],
    }, indent=2))


if __name__ == "__main__":
    main()
