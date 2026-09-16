#!/usr/bin/env python3
"""Word alignment infrastructure (word-level stage 0).

No model representations are computed here. This module establishes, and
measures the reliability of, the word correspondences that a word-level
decomposition would rest on. If the correspondences are not trustworthy,
nothing measured on top of them is.

DESIGN
  Pivot. Every alignment is English to target, matching how NTREX itself was
  built (English source, professional translation into each language). No
  target-to-target alignment is attempted.

  Concept side. "Content word" is defined on the ENGLISH side only, which is
  the side that will supply concept identity downstream. One stopword list is
  therefore sufficient and no per-language lexical resources are needed.
  Function words are counted and reported separately rather than discarded,
  because their alignment behaviour is expected to differ and that difference
  is itself informative.

  Two independent aligners. A neural one (SimAlign over XLM-R embeddings,
  no fine-tuning) and a statistical one (eflomal, IBM-model style, trained
  unsupervised on the parallel text itself). They share no machinery, so
  their agreement is a meaningful reliability estimate rather than two views
  of the same model.

  Substitution note. The plan named awesome-align as the neural aligner.
  SimAlign is used instead: same underlying idea (contextual embeddings from
  XLM-R, no fine-tuning) and a published method, but installable as a package
  rather than a research repository pinned to an old transformers version.
  This is an implementation decision forced by the environment, recorded here
  so it is visible rather than silent.

KNOWN LIMITATION, reported rather than worked around. Word segmentation is
whitespace based. For languages written without whitespace word boundaries
(Chinese, Japanese, Khmer, Burmese, Thai) this is wrong, and the coverage
table will show it as degenerate token counts. Those languages need a
segmenter before any word-level claim can include them. The coverage table
is the evidence for that statement.

Usage:
    python -m src.wordlevel.align --languages configs/wordlevel/prototype20.yaml
Outputs: results/wordlevel/alignments/<lang>.json, plus coverage.json
"""
import argparse
import json
import os
import re
import string
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.common import ledger  # noqa: E402

NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')
OUT = os.path.join(ROOT, 'results', 'wordlevel', 'alignments')

# English function words. Content is defined as the complement, on the pivot
# side only. Deliberately a plain closed-class list rather than a tagger, so
# the definition is inspectable and has no model dependency.
STOPWORDS = set("""
a an the this that these those and or but nor for yet so if then than as
because while when where which who whom whose what why how all any both each
few more most other some such no not only own same too very can will just
don should now i you he she it we they me him her us them my your his its our
their mine yours hers ours theirs am is are was were be been being have has
had having do does did doing would could shall may might must of in on at by
to from up down out off over under again further once here there about
above below between into through during before after with without within
against among across behind beyond
""".split())

PUNCT = set(string.punctuation) | set('“”‘’—–…«»„‟‹›¡¿·।॥、。，；：？！（）《》')


def is_content(token):
    """English-side content word: not punctuation, not a numeral, not a
    function word, and containing at least one alphabetic character."""
    t = token.strip(string.punctuation + '“”‘’—–…')
    if not t or not any(c.isalpha() for c in t):
        return False
    if re.fullmatch(r'[\d.,:/%-]+', t):
        return False
    return t.lower() not in STOPWORDS


# Segmenters for scripts written without whitespace word boundaries. Each is
# the standard tool for its language. Loaded lazily and cached, since some
# carry model files.
#
# IMPORTANT CAVEAT, to be carried into any result that uses these. For these
# languages the word is not an orthographic fact but a segmenter's decision,
# and different segmenters disagree. A word-level unit for Chinese is
# therefore not the same kind of object as a word-level unit for German. The
# segmenter used is recorded per language in the output so that any claim can
# be traced to the tool that made the units.
_SEGMENTERS = {}


def _segmenter(lang):
    if lang in _SEGMENTERS:
        return _SEGMENTERS[lang]
    fn = None
    try:
        if lang.startswith('zho') or lang.startswith('yue'):
            import jieba
            # jieba writes a cache on first use; the login node's root
            # filesystem is full, so point it into our own tmp
            tmp = os.environ.get('TMPDIR')
            if tmp:
                jieba.dt.tmp_dir = tmp
            fn = ('jieba', lambda s: list(jieba.cut(s)))
        elif lang.startswith('jpn'):
            import fugashi
            tagger = fugashi.Tagger()
            fn = ('fugashi+unidic-lite',
                  lambda s: [w.surface for w in tagger(s)])
        elif lang.startswith('khm'):
            from khmernltk import word_tokenize as kwt
            fn = ('khmer-nltk', kwt)
    except Exception as e:                       # tool missing or unusable
        print(f'    [segmenter] {lang}: unavailable ({type(e).__name__}), '
              f'falling back to whitespace', flush=True)
        fn = None
    _SEGMENTERS[lang] = fn
    return fn


def tokenize(line, lang=None):
    """Word tokenization.

    Whitespace with edge punctuation stripped, which is correct for scripts
    that mark word boundaries. For scripts that do not (Han, Japanese, Khmer,
    Burmese) whitespace returns whole clauses as single tokens, so a
    language-specific segmenter is used where a standard one exists. Where
    none exists the whitespace result is returned and the coverage table
    shows the consequence rather than hiding it."""
    seg = _segmenter(lang) if lang else None
    if seg is not None:
        pieces = [p for p in seg[1](line) if p.strip()]
    else:
        pieces = line.split()
    out = []
    for tok in pieces:
        core = tok.strip(''.join(PUNCT))
        if core:
            out.append(core)
    return out


def segmenter_name(lang):
    seg = _segmenter(lang) if lang else None
    return seg[0] if seg is not None else 'whitespace'


def load_pair(lang, n_sents, n_train):
    """English and target sentences. `n_train` sentences are used to fit the
    statistical aligner (unsupervised, on the parallel text itself, so using
    all available text is standard and involves no labels); the first
    `n_sents` are the evaluation subset."""
    src_p = os.path.join(NTREX, 'newstest2019-src.eng.txt')
    tgt_p = os.path.join(NTREX, f'newstest2019-ref.{lang}.txt')
    if not os.path.exists(tgt_p):
        return None
    with open(src_p) as f:
        eng = [l.strip() for l in f]
    with open(tgt_p) as f:
        tgt = [l.strip() for l in f]
    n = min(len(eng), len(tgt), n_train)
    pairs = [(tokenize(e, 'eng'), tokenize(t, lang))
             for e, t in zip(eng[:n], tgt[:n]) if e.strip() and t.strip()]
    return pairs, min(n_sents, len(pairs))


def run_simalign(pairs, n_eval, model_name='xlm-roberta-base', device=None):
    """Neural alignment. Returns per-sentence link sets for the intersection
    method (high precision) and itermax (higher recall). Device only affects
    speed: the alignment is a deterministic function of the embeddings."""
    from simalign import SentenceAligner
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    aligner = SentenceAligner(model=model_name, token_type='bpe',
                              matching_methods='ai', device=device)
    inter, itermax = [], []
    for i, (s, t) in enumerate(pairs[:n_eval]):
        if not s or not t:
            inter.append(set()); itermax.append(set()); continue
        a = aligner.get_word_aligns(s, t)
        inter.append(set(map(tuple, a.get('inter', []))))
        itermax.append(set(map(tuple, a.get('itermax', []))))
        if (i + 1) % 50 == 0:
            print(f'    simalign {i + 1}/{n_eval}', flush=True)
    return inter, itermax


def run_eflomal(pairs, n_eval):
    """Statistical alignment. Fitted on all supplied pairs, links read back
    for the evaluation subset. Symmetrized by intersecting the forward and
    reverse directions, which is the high-precision choice and matches the
    role this aligner plays here as an independent check."""
    import eflomal
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as td:
        sp, tp = os.path.join(td, 'src'), os.path.join(td, 'tgt')
        with open(sp, 'w') as fs, open(tp, 'w') as ft:
            for s, t in pairs:
                fs.write(' '.join(s) + '\n')
                ft.write(' '.join(t) + '\n')
        fwd, rev = os.path.join(td, 'fwd'), os.path.join(td, 'rev')
        aligner = eflomal.Aligner()
        with open(sp) as fs, open(tp) as ft:
            aligner.align(fs, ft, links_filename_fwd=fwd,
                          links_filename_rev=rev, quiet=True)

        def read(path):
            out = []
            with open(path) as f:
                for line in f:
                    links = set()
                    for tok in line.split():
                        if '-' in tok:
                            i, j = tok.split('-')
                            links.add((int(i), int(j)))
                    out.append(links)
            return out

        F, R = read(fwd), read(rev)
    return [F[i] & R[i] for i in range(min(n_eval, len(F), len(R)))]


def one_to_one(links):
    """Keep only links where the English token and the target token each
    appear exactly once. This is the headline set: unambiguous
    correspondences."""
    from collections import Counter
    ls, lt = Counter(i for i, _ in links), Counter(j for _, j in links)
    return {(i, j) for i, j in links if ls[i] == 1 and lt[j] == 1}


def agreement(a, b):
    """Link-set agreement between two aligners over a corpus: intersection
    over union, plus F1 with the first treated as reference."""
    inter = sum(len(x & y) for x, y in zip(a, b))
    na, nb = sum(map(len, a)), sum(map(len, b))
    union = na + nb - inter
    return {
        'iou': round(inter / union, 4) if union else 0.0,
        'f1': round(2 * inter / (na + nb), 4) if (na + nb) else 0.0,
        'n_links_a': na, 'n_links_b': nb, 'n_shared': inter,
    }


def coverage(pairs, links, n_eval):
    """Fraction of English content tokens receiving a link, and the same for
    function tokens. Reported separately, per the design note."""
    c_tot = c_cov = f_tot = f_cov = 0
    src_toks = tgt_toks = 0
    for (s, t), L in zip(pairs[:n_eval], links):
        aligned = {i for i, _ in L}
        src_toks += len(s); tgt_toks += len(t)
        for i, tok in enumerate(s):
            if is_content(tok):
                c_tot += 1; c_cov += (i in aligned)
            else:
                f_tot += 1; f_cov += (i in aligned)
    return {
        'content_tokens': c_tot,
        'content_aligned': c_cov,
        'content_coverage': round(c_cov / c_tot, 4) if c_tot else 0.0,
        'function_tokens': f_tot,
        'function_coverage': round(f_cov / f_tot, 4) if f_tot else 0.0,
        'mean_src_tokens_per_sent': round(src_toks / max(n_eval, 1), 2),
        'mean_tgt_tokens_per_sent': round(tgt_toks / max(n_eval, 1), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--languages', required=True,
                    help='yaml with a `languages` list and tier labels')
    ap.add_argument('--n_sents', type=int, default=300)
    ap.add_argument('--n_train', type=int, default=2000,
                    help='sentences used to fit the statistical aligner')
    ap.add_argument('--only', default=None, help='comma-separated subset')
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.languages))
    langs = list(cfg['languages'])
    if args.only:
        keep = set(args.only.split(','))
        langs = [l for l in langs if l in keep]

    os.makedirs(OUT, exist_ok=True)
    rec = ledger.open_run('wordlevel_align',
                          {'n_sents': args.n_sents, 'n_train': args.n_train,
                           'languages': langs})
    try:
        summary = {}
        for lang in langs:
            print(f'[{lang}]', flush=True)
            loaded = load_pair(lang, args.n_sents, args.n_train)
            if loaded is None:
                print('  no NTREX file, skipped'); continue
            pairs, n_eval = loaded
            sim_inter, sim_itermax = run_simalign(pairs, n_eval)
            efl = run_eflomal(pairs, n_eval)
            n = min(len(sim_inter), len(efl))

            # Headline link set. The stage gate failed at twenty languages
            # (2 of 7 high-resource pairs above 0.70 agreement), so rather
            # than trusting either aligner alone we keep only links that
            # BOTH produce and that are one-to-one on each side. Reliability
            # is bought with coverage, and the coverage cost is reported per
            # language rather than absorbed silently.
            head = [one_to_one(x) for x in sim_inter[:n]]
            agreed = [one_to_one(sim_inter[i] & efl[i]) for i in range(n)]
            row = {
                'language': lang,
                'tier': cfg.get('tiers', {}).get(lang),
                'script': cfg.get('scripts', {}).get(lang),
                'segmenter': segmenter_name(lang),
                'n_sentences': n,
                'agreement_simalign_inter_vs_eflomal':
                    agreement(sim_inter[:n], efl[:n]),
                'agreement_simalign_itermax_vs_eflomal':
                    agreement(sim_itermax[:n], efl[:n]),
                'coverage_headline_one_to_one': coverage(pairs, head, n),
                'coverage_simalign_itermax': coverage(pairs, sim_itermax[:n], n),
                'coverage_agreed_one_to_one': coverage(pairs, agreed, n),
                'n_links_agreed': int(sum(map(len, agreed))),
                'n_links_simalign_only': int(sum(map(len, head))),
            }
            summary[lang] = row
            with open(os.path.join(OUT, f'{lang}.json'), 'w') as f:
                json.dump({**row,
                           'links_one_to_one': [sorted(x) for x in head],
                           'links_agreed': [sorted(x) for x in agreed],
                           'sentences_src': [s for s, _ in pairs[:n]],
                           'sentences_tgt': [t for _, t in pairs[:n]]}, f)
            a = row['agreement_simalign_inter_vs_eflomal']
            c = row['coverage_headline_one_to_one']
            ag = row['coverage_agreed_one_to_one']
            print(f"  agreement IoU {a['iou']:.3f} F1 {a['f1']:.3f} | "
                  f"content coverage {c['content_coverage']:.3f} "
                  f"-> agreed {ag['content_coverage']:.3f} "
                  f"({row['n_links_agreed']} links) | "
                  f"tokens/sent en {c['mean_src_tokens_per_sent']:.1f} "
                  f"tgt {c['mean_tgt_tokens_per_sent']:.1f}", flush=True)

        with open(os.path.join(ROOT, 'results', 'wordlevel',
                               'coverage.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        ledger.close_run(rec, 'done', {'n_languages': len(summary)})
        print(f'[done] {len(summary)} languages -> results/wordlevel/')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
