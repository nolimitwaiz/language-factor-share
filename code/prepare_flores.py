#!/usr/bin/env python3
"""E-B data preparation: download FLORES-200, map NTREX-128 language codes to FLORES-200 codes,
and write the devtest split in the file layout cluster_grid_corpus.py reads.

Writes data/FLORES200/flores200_dataset/ (extracted tarball),
       data/FLORES200/devtest_ntrex_layout/flores200-devtest.<ntrex_code>.txt (+ .eng.txt),
       data/FLORES200/ntrex_to_flores.json (the frozen mapping, with unmapped codes listed).
"""
import json, os, sys, tarfile, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX = os.path.join(ROOT, "data", "NTREX", "NTREX-128")
OUT = os.path.join(ROOT, "data", "FLORES200")
URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# explicit NTREX -> FLORES-200 mappings where the 3-letter prefix is ambiguous or differs
EXPLICIT = {
    "arb": "arb_Arab", "aze-Latn": "azj_Latn", "ckb-Arab": "ckb_Arab", "zho-CN": "zho_Hans", "zho-TW": "zho_Hant",
    "srp-Cyrl": "srp_Cyrl", "fas": "pes_Arab", "prs": "prs_Arab", "msa": "zsm_Latn", "fil": "tgl_Latn",
    "swa": "swh_Latn", "mon": "khk_Cyrl", "sqi": "als_Latn", "nor": "nob_Latn", "kmr": "kmr_Latn",
    "orm": "gaz_Latn", "que": "quy_Latn", "uzb": "uzn_Latn", "lav": "lvs_Latn", "est": "est_Latn",
    "pus": "pbt_Arab", "yue": "yue_Hant", "tir": "tir_Ethi", "nep": "npi_Deva", "ory": "ory_Orya",
    "mlg": "plt_Latn", "grn": "grn_Latn", "kur": "kmr_Latn", "aym": "ayr_Latn", "div": "div_Thaa",
    "fuc": "fuv_Latn", "mey": None, "hmn": None, "eng-GB": None, "eng-IN": None, "eng-US": None, "fra-CA": None,
    "spa-MX": "spa_Latn", "por-BR": "por_Latn", "ptb": "por_Latn",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    tgz = os.path.join(OUT, "flores200_dataset.tar.gz")
    if not os.path.exists(tgz):
        print("[download]", URL, flush=True)
        urllib.request.urlretrieve(URL, tgz)
    ds = os.path.join(OUT, "flores200_dataset")
    if not os.path.isdir(ds):
        with tarfile.open(tgz) as t:
            t.extractall(OUT)
    devtest = os.path.join(ds, "devtest")
    flores_codes = sorted(f[:-len(".devtest")] for f in os.listdir(devtest) if f.endswith(".devtest"))
    by_prefix = {}
    for c in flores_codes:
        by_prefix.setdefault(c[:3], []).append(c)
    ntrex_codes = sorted(f[len("newstest2019-ref."):-4] for f in os.listdir(NTREX) if f.startswith("newstest2019-ref."))
    mapping, unmapped, ambiguous = {}, [], {}
    for code in ntrex_codes:
        if code in EXPLICIT:
            if EXPLICIT[code] is None:
                unmapped.append(code)
            else:
                mapping[code] = EXPLICIT[code]
            continue
        cands = by_prefix.get(code[:3], [])
        if len(cands) == 1:
            mapping[code] = cands[0]
        elif len(cands) == 0:
            unmapped.append(code)
        else:
            ambiguous[code] = cands
            unmapped.append(code)
    for code, fl in list(mapping.items()):
        if fl not in flores_codes:
            print("[warn] mapped code not in FLORES:", code, fl); unmapped.append(code); del mapping[code]
    lay = os.path.join(OUT, "devtest_ntrex_layout")
    os.makedirs(lay, exist_ok=True)
    def read(fl):
        return [l.rstrip("\n") for l in open(os.path.join(devtest, fl + ".devtest"), encoding="utf-8")]
    eng = read("eng_Latn")
    open(os.path.join(lay, "flores200-devtest.eng.txt"), "w", encoding="utf-8").write("\n".join(eng) + "\n")
    n_written = 0
    for code, fl in sorted(mapping.items()):
        if code == "eng":
            continue
        lines = read(fl)
        assert len(lines) == len(eng), (code, fl, len(lines), len(eng))
        open(os.path.join(lay, f"flores200-devtest.{code}.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        n_written += 1
    report = {"n_ntrex_codes": len(ntrex_codes), "n_mapped": len([c for c in mapping if c != "eng"]),
              "n_written_non_english": n_written, "n_sentences": len(eng), "unmapped": sorted(set(unmapped)),
              "ambiguous_prefixes": ambiguous, "mapping": mapping}
    json.dump(report, open(os.path.join(OUT, "ntrex_to_flores.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "mapping"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
