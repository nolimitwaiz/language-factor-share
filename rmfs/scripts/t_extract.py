#!/usr/bin/env python
"""Behavioral-T extraction (Phase 2.2). GPU, cluster-only, one model or arm
checkpoint per invocation. Self-contained at the n=300 development protocol:
extracts its own pooled embeddings, fits its own basis/GPA/reverse maps on
document-purged TRAIN concepts, and measures the five NLL conditions on
pairs drawn from HELD-OUT concepts only.

Frozen governance: prereg §3.1 (T formula, ε_B), addendum A2 (reverse-
direction rung maps, saturation), addendum A3 (leave-one-out consensus as
primary, factor-input reconstruction, broadcast strict / hybrid
sensitivity, arms inherit the base model's rung selection).

Conditions per (language ℓ, pair): none / native / mismatch (wrong
document) / recon-strict / recon-hybrid. Reconstruction replaces the
context token states at the dip layer via a forward hook; the hybrid
condition first captures the original states in a no-hook pass.

Usage:
  python scripts/t_extract.py --model_tag Qwen3-0.6B-Base
  python scripts/t_extract.py --arm_ckpt ckpt_armWA-C_seed0
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys

import numpy as np

RMFS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RMFS, "src"))
PRIOR = os.path.expanduser("~/multilingual-metrics")
NTREX = os.path.join(PRIOR, "data", "NTREX", "NTREX-128")
DOC_IDS = os.path.join(PRIOR, "data", "NTREX", "DOCUMENT_IDS.tsv")
CKPTS = os.path.join(PRIOR, "results", "aim2")

from rmfs.components.ladder import consensus_target, gpa  # noqa: E402
from rmfs.components.transfer import (  # noqa: E402
    fit_reverse_map,
    reconstruct,
)
from rmfs.data.ntrex import load_ntrex, purged_concept_splits  # noqa: E402

N_SENTS = 300
BASE_TAG = "Qwen3-0.6B-Base"
# the legacy 3.1.3 language set (NTREX codes), eng is pivot/target
CTX_LANGS = ["deu", "fra", "spa", "por", "ita", "nld", "swe", "pol", "ces",
             "ron", "rus", "ukr", "bul", "ell", "tur", "arb", "heb", "fas",
             "hin", "ben", "tam", "tel", "mar", "urd", "vie", "ind", "msa",
             "tha", "zho-CN", "jpn", "kor", "fin", "hun", "est", "kat",
             "amh", "swa", "yor", "khm", "mya"]

MODEL_REGISTRY = os.path.join(RMFS, "configs", "models.yaml")


def load_registry():
    import yaml

    return yaml.safe_load(open(MODEL_REGISTRY))["models"]


def rungs_for(tag: str) -> dict:
    p = os.path.join(RMFS, "results", "runs", f"grid_components_{tag}",
                     "result.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no Phase-1 rung selection for {tag} at {p} — models without "
            f"one run their own labeled selection task (A3 §4); refusing "
            f"to guess")
    d = json.load(open(p))
    return {lang: v["rung"] for lang, v in d["K"].items()}


def pooled_embed(model, tok, sentences, dev, dip, batch=16, max_len=128):
    import torch

    out = []
    for i in range(0, len(sentences), batch):
        enc = tok(sentences[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(dev)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hs[dip].float() * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out, 0)


def make_pairs(doc_ids, test_idx, n_pairs=60):
    """Adjacent same-document pairs (i context, i+1 target) with BOTH
    sentences in the held-out concept set; wrong-document mismatch partner
    for each. Purge-consistent: no pair concept was seen by any map fit."""
    test = set(int(i) for i in test_idx)
    pairs = []
    for i in range(N_SENTS - 1):
        if i in test and (i + 1) in test and doc_ids[i] == doc_ids[i + 1]:
            pairs.append(i)
        if len(pairs) >= n_pairs:
            break
    mis = []
    for k, i in enumerate(pairs):
        j = pairs[(k + len(pairs) // 2) % len(pairs)]
        if doc_ids[j] == doc_ids[i]:
            j = pairs[(k + len(pairs) // 2 + 1) % len(pairs)]
        mis.append(j)
    return pairs, mis


def scaffold_ids(tok, n_ctx: int):
    """Neutral scaffold: the separator token repeated to match the native
    context's token count (A4 — no new free parameter)."""
    import torch

    nl = tok("\n", add_special_tokens=False).input_ids
    unit = nl[-1] if nl else (tok.eos_token_id or 0)
    return torch.tensor([[unit] * n_ctx])


def nll_of_target(model, tok, dev, dip, target: str,
                  context: str | None = None,
                  context_ids=None,
                  replace_states: np.ndarray | None = None,
                  capture_only: bool = False):
    """Mean NLL of target tokens. With `replace_states`, a forward hook
    substitutes the context-position states at the dip layer. With
    `capture_only`, returns the dip-layer context states instead."""
    import torch

    tgt = tok(" " + target, return_tensors="pt", add_special_tokens=False)
    if context_ids is not None:
        ids = torch.cat([context_ids, tgt.input_ids], 1).to(dev)
        n_ctx = context_ids.shape[1]
    elif context is not None:
        ctx = tok(context + "\n", return_tensors="pt")
        ids = torch.cat([ctx.input_ids, tgt.input_ids], 1).to(dev)
        n_ctx = ctx.input_ids.shape[1]
    else:
        bos = tok.bos_token_id or tok.eos_token_id
        ids = torch.cat([torch.tensor([[bos]]), tgt.input_ids], 1).to(dev)
        n_ctx = 1

    # hidden_states[dip] = output of block dip-1 (index 0 = embeddings).
    # Architecture-aware block lookup: Qwen/OLMo/Llama keep blocks at
    # model.model.layers; BLOOM at model.transformer.h. Hard-fail on
    # anything else rather than hooking the wrong module.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    else:
        raise AttributeError(
            f"unknown block layout for {type(model).__name__}; refusing "
            f"to guess a hook point")
    layer = blocks[dip - 1]
    captured = {}
    hook = None
    if capture_only or replace_states is not None:
        if replace_states is not None:
            rep = torch.tensor(replace_states, dtype=next(
                model.parameters()).dtype, device=dev)

        def _hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if capture_only:
                captured["states"] = h[0, :n_ctx, :].float().cpu().numpy()
                return out
            h = h.clone()
            h[0, :n_ctx, :] = rep[:n_ctx]
            return (h, *out[1:]) if isinstance(out, tuple) else h

        hook = layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits = model(ids).logits.float()
    finally:
        if hook is not None:
            hook.remove()
    if capture_only:
        return captured["states"]
    logp = torch.log_softmax(logits[0, :-1], -1)
    tgt_ids = ids[0, 1:]
    return float(-(logp[torch.arange(len(tgt_ids)), tgt_ids][n_ctx - 1:])
                 .mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag")
    ap.add_argument("--arm_ckpt")
    ap.add_argument("--n_pairs", type=int, default=60)
    ap.add_argument("--langs", type=int, default=len(CTX_LANGS))
    ap.add_argument("--diagnostic", action="store_true",
                    help="instrument validation: identity-patch (captured "
                         "originals fed back through the hook - must equal "
                         "native to float noise or the harness is broken), "
                         "mean-patch, and sink-preserving / norm-matched "
                         "reconstruction variants. No T is computed.")
    args = ap.parse_args()
    if bool(args.model_tag) == bool(args.arm_ckpt):
        raise SystemExit("exactly one of --model_tag / --arm_ckpt")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reg = load_registry()
    if args.arm_ckpt:
        run_name = args.arm_ckpt
        model_path = os.path.join(CKPTS, args.arm_ckpt)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"checkpoint dir missing: {model_path}")
        base = reg[BASE_TAG]
        dip, rank = base["dip_layer"], 64
        rung_by_lang = rungs_for(BASE_TAG)          # A3 §3: arms inherit
    else:
        run_name = args.model_tag
        m = reg[args.model_tag]
        model_path = m["hf_id"]
        dip = m["dip_layer"]
        gp = json.load(open(os.path.join(
            RMFS, "results", "runs", f"grid_components_{args.model_tag}",
            "result.json")))
        rank = gp["selected_rank_reused"]
        rung_by_lang = rungs_for(args.model_tag)

    dev = "cuda"
    dtype = torch.bfloat16                          # a100; bloom-safe
    tok = AutoTokenizer.from_pretrained(
        model_path if args.arm_ckpt else model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype).to(dev).eval()
    print(f"[{run_name}] dip L{dip}, rank {rank}, dtype bf16", flush=True)

    langs = [lg for lg in CTX_LANGS[: args.langs]]
    data = load_ntrex(NTREX, DOC_IDS, langs, N_SENTS)
    doc_ids = list(data.doc_ids)

    # ---- pooled embeddings at the dip layer, fp32 pooling
    E = {}
    for lg in ["eng", *langs]:
        E[lg] = pooled_embed(model, tok, data.sentences[lg], dev, dip)
        print(f"  [embed] {lg}", flush=True)

    # ---- basis, mean, purged split, GPA on train
    all_langs = ["eng", *langs]
    flat = np.concatenate([E[lg] for lg in all_langs], 0).astype(np.float32)
    mu = flat.mean(0).astype(np.float64)
    Cc = ((flat - mu.astype(np.float32)).T
          @ (flat - mu.astype(np.float32))).astype(np.float64)
    _, evecs = np.linalg.eigh(Cc)
    basis = evecs[:, ::-1][:, :rank]
    Zc = {lg: ((E[lg] - mu) @ basis) for lg in all_langs}

    tr, te = purged_concept_splits(doc_ids, n_splits=1, seed=13)[0]
    g = gpa({lg: Zc[lg][tr] for lg in all_langs})
    print(f"[gpa] train {len(tr)} / held-out {len(te)} concepts", flush=True)

    pairs, mis = make_pairs(doc_ids, te, args.n_pairs)
    print(f"[pairs] {len(pairs)} held-out same-document pairs", flush=True)

    rows = []
    eng = data.sentences["eng"]
    for lg in langs:
        rung = rung_by_lang.get(lg)
        if rung is None:
            print(f"  [skip] {lg}: no rung selection", flush=True)
            continue
        # LOO consensus coords (exclude=lg), train for fitting, test for use
        z_tr = consensus_target(g, {l2: Zc[l2][tr] for l2 in all_langs},
                                exclude=lg)
        z_te = consensus_target(g, {l2: Zc[l2][te] for l2 in all_langs},
                                exclude=lg)
        rev = fit_reverse_map(z_tr, Zc[lg][tr], rung, seed=13)
        te_pos = {int(c): k for k, c in enumerate(te)}
        sents = data.sentences[lg]
        if args.diagnostic:
            for k in range(min(len(pairs), 4)):
                i = pairs[k]
                target = eng[i + 1]
                nll_nat = nll_of_target(model, tok, dev, dip, target,
                                        context=sents[i])
                orig = nll_of_target(model, tok, dev, dip, target,
                                     context=sents[i], capture_only=True)
                nll_id = nll_of_target(model, tok, dev, dip, target,
                                       context=sents[i],
                                       replace_states=orig)
                mu_patch = np.repeat(mu[None, :], orig.shape[0], axis=0)
                nll_mu = nll_of_target(model, tok, dev, dip, target,
                                       context=sents[i],
                                       replace_states=mu_patch)
                z_i = z_te[te_pos[i]]
                rec = reconstruct(z_i, basis, mu, rev, mode="strict",
                                  n_tokens=orig.shape[0])
                # sink-preserving: keep position 0 original
                rec_sink = rec.copy()
                rec_sink[0] = orig[0]
                nll_sink = nll_of_target(model, tok, dev, dip, target,
                                         context=sents[i],
                                         replace_states=rec_sink)
                # norm-matched + sink-preserving
                norms_o = np.linalg.norm(orig, axis=1, keepdims=True)
                norms_r = np.linalg.norm(rec, axis=1, keepdims=True) + 1e-9
                rec_nm = rec * (norms_o / norms_r)
                rec_nm[0] = orig[0]
                nll_nm = nll_of_target(model, tok, dev, dip, target,
                                       context=sents[i],
                                       replace_states=rec_nm)
                rows.append({"lang": lg, "pair": i, "rung": rung,
                             "nll_native": nll_nat, "nll_identity": nll_id,
                             "nll_mean_patch": nll_mu,
                             "nll_strict": float("nan"),
                             "nll_sink_preserved": nll_sink,
                             "nll_norm_matched_sink": nll_nm})
                print(f"  [{lg} p{i}] native {nll_nat:.3f} | identity "
                      f"{nll_id:.3f} (delta {nll_id-nll_nat:+.4f}) | "
                      f"mean-patch {nll_mu:.3f} | sink-kept {nll_sink:.3f}"
                      f" | norm+sink {nll_nm:.3f}", flush=True)
            continue
        n_udf = n_sat = 0
        for k in range(len(pairs)):
            i, j = pairs[k], mis[k]                 # index loop: local dev
            target = eng[i + 1]                     # py3.9 lacks zip(strict=)
            nll_none = nll_of_target(model, tok, dev, dip, target)
            nll_nat = nll_of_target(model, tok, dev, dip, target,
                                    context=sents[i])
            nll_mis = nll_of_target(model, tok, dev, dip, target,
                                    context=sents[j])
            # A4 injection conditions: neutral scaffold, factor states
            n_ctx = tok(sents[i] + "\n",
                        return_tensors="pt").input_ids.shape[1]
            scaf = scaffold_ids(tok, n_ctx)
            z_i = z_te[te_pos[i]]
            z_j = z_te[te_pos[j]]
            rec_i = reconstruct(z_i, basis, mu, rev, mode="strict",
                                n_tokens=n_ctx)
            rec_j = reconstruct(z_j, basis, mu, rev, mode="strict",
                                n_tokens=n_ctx)
            nll_scaf = nll_of_target(model, tok, dev, dip, target,
                                     context_ids=scaf)
            nll_inj = nll_of_target(model, tok, dev, dip, target,
                                    context_ids=scaf, replace_states=rec_i)
            nll_injw = nll_of_target(model, tok, dev, dip, target,
                                     context_ids=scaf, replace_states=rec_j)
            denom = ((nll_none - nll_nat) - (nll_none - nll_mis))
            undefined = bool(denom < 0.072)
            saturated = bool((not undefined)
                             and (nll_inj > nll_injw + 0.072))
            t_inj = (float("nan") if undefined
                     else (nll_injw - nll_inj) / denom)
            n_udf += int(undefined)
            n_sat += int(saturated)
            rows.append({
                "lang": lg, "pair": i, "rung": rung,
                "nll_none": nll_none, "nll_native": nll_nat,
                "nll_mismatch": nll_mis, "nll_scaffold": nll_scaf,
                "nll_inject": nll_inj, "nll_inject_wrong": nll_injw,
                "t_inj": t_inj,
                "q_t": (float("nan") if undefined
                        else float(np.clip(t_inj, 0.0, 1.0))),
                "undefined": undefined, "saturated": saturated,
            })
        valid = [r for r in rows if r["lang"] == lg
                 and not (r["undefined"] or r["saturated"])]
        mt = (float(np.mean([r["t_inj"] for r in valid]))
              if valid else float("nan"))
        print(f"  [{lg}] rung {rung}: T_inj {mt:+.3f} over "
              f"{len(valid)} valid ({n_udf} undefined, {n_sat} saturated)",
              flush=True)

    run_id = f"t_extract_{run_name}"
    out_dir = os.path.join(RMFS, "results", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "t_rows.parquet"))
    git = subprocess.run(["git", "-C", RMFS, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    manifest = {"run_id": run_id, "git": git, "host": socket.gethostname(),
                "when": datetime.datetime.now().isoformat(timespec="seconds"),
                "config": {**vars(args), "dip": dip, "rank": rank,
                           "n_train": int(len(tr)), "n_test": int(len(te)),
                           "n_pairs": len(pairs), "dtype": "bf16",
                           "consensus": "leave-one-out (A3)",
                           "governance": ["prereg", "A2", "A3", "A4-injection"]},
                "inputs": {"model": str(model_path),
                           "rungs_from": (BASE_TAG if args.arm_ckpt
                                          else run_name)}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[done] {len(rows)} rows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
