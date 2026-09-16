#!/usr/bin/env python3
"""Code-switched data generation from parallel text (proposal section 3.2.1).

The proposal: "Code-switched text is a continuous narrative in which the
language switches on a word-by-word, phrase-by-phrase, or sentence-by-
sentence basis... We can create such data from parallel text."

This builds it at the word and phrase level using the word alignments
produced for Aim 1, which means every substitution is grounded in a
correspondence that two independent aligners agreed on, and the reliability
of that correspondence is known per language.

WHY THIS IS NOT A NEUTRAL UTILITY. Stage 0 measured that word alignment
reliability tracks typology, not resource level: inter-aligner agreement
runs 0.82 for Indonesian down to 0.33 for Zulu and 0.13 for Burmese. Word
substitution in a language whose alignments are unreliable produces
mislabelled training data, silently. Every generated corpus therefore
carries its per-language alignment agreement in the manifest, and languages
below a declared threshold are excluded by default rather than degraded
quietly.

Ground truth is retained for every switch: which source token was replaced,
by what, and from which language. That makes the generated data usable as
supervision as well as as input.

Usage:
    python -m src.aim2.codeswitch --rates 0.1 0.25 0.5 --min_agreement 0.55
Outputs: data/codeswitch/<rate>/corpus.jsonl + manifest.json
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.common import ledger  # noqa: E402

ALIGN = os.path.join(ROOT, 'results', 'wordlevel', 'alignments')
OUT = os.path.join(ROOT, 'data', 'codeswitch')

# Sentence ranges. The study must never train on text used to measure
# anything, so the training range is disjoint from the evaluation range used
# throughout Aim 1 (the first 300 sentences) and from the word-level
# analyses (which used the full corpus for alignment but report on the
# first 300).
TRAIN_RANGE = (500, 1900)


def load_language(lang):
    p = os.path.join(ALIGN, f'{lang}.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        'links': d['links_agreed'],
        'src': d['sentences_src'],
        'tgt': d['sentences_tgt'],
        'agreement': d['agreement_simalign_inter_vs_eflomal']['f1'],
        'coverage': d['coverage_agreed_one_to_one']['content_coverage'],
        'segmenter': d.get('segmenter', 'whitespace'),
        'tier': d.get('tier'),
        'script': d.get('script'),
    }


def switch_sentence(src_words, tgt_words, links, rate, rng, mode='word',
                    drop_identity=True):
    """Replace a fraction of aligned English words with their counterparts.

    mode='word'   independent per-token substitution
    mode='phrase' substitutes contiguous runs, which is closer to how
                  code-switching occurs naturally and produces a different
                  distribution of switch-point contexts

    drop_identity removes candidates whose target form is identical to the
    source, which are not code-switches at all. Measured before this filter
    existed: 11 percent of all substitutions overall, and up to 17 percent
    for Latin-script languages (Swahili, German, Vietnamese, Indonesian),
    against far fewer for non-Latin scripts. Leaving them in would give
    Latin-script languages systematically fewer real switches than
    Devanagari or Han languages at the same nominal rate, which is a
    script-correlated confound in the training data rather than a nuisance.

    Returns (tokens, switches) where switches records the ground truth."""
    if not links:
        return None
    amap = dict(links)
    aligned = sorted(amap)
    if drop_identity:
        aligned = [i for i in aligned
                   if amap[i] < len(tgt_words) and i < len(src_words)
                   and src_words[i].lower() != tgt_words[amap[i]].lower()]
        if not aligned:
            return None
    n_switch = int(round(len(aligned) * rate))
    if n_switch < 1:
        return None

    if mode == 'phrase':
        chosen, span = set(), max(2, int(round(1 / max(rate, 1e-6) ** 0.5)))
        starts = rng.sample(aligned, k=min(len(aligned),
                                           max(1, n_switch // span + 1)))
        for st in starts:
            i = aligned.index(st)
            for j in range(i, min(i + span, len(aligned))):
                if len(chosen) < n_switch:
                    chosen.add(aligned[j])
    else:
        chosen = set(rng.sample(aligned, k=min(n_switch, len(aligned))))

    out, switches = [], []
    for i, w in enumerate(src_words):
        if i in chosen and amap[i] < len(tgt_words):
            out.append(tgt_words[amap[i]])
            switches.append({'pos': i, 'source': w,
                             'replacement': tgt_words[amap[i]]})
        else:
            out.append(w)
    return out, switches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[0.1, 0.25, 0.5])
    ap.add_argument('--min_agreement', type=float, default=0.55,
                    help='exclude languages whose two aligners agree less '
                         'than this; measured in Stage 0')
    ap.add_argument('--modes', nargs='+', default=['word', 'phrase'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep_identity', action='store_true',
                    help='keep substitutions where the target form equals '
                         'the source; off by default, see switch_sentence')
    args = ap.parse_args()

    rec = ledger.open_run('codeswitch_generate', vars(args), seed=args.seed)
    try:
        langs = sorted(f[:-5] for f in os.listdir(ALIGN) if f.endswith('.json'))
        loaded, excluded = {}, {}
        for lg in langs:
            d = load_language(lg)
            if d is None:
                continue
            if d['agreement'] < args.min_agreement:
                excluded[lg] = round(d['agreement'], 3)
            else:
                loaded[lg] = d
        print(f'[languages] {len(loaded)} usable, {len(excluded)} excluded '
              f'for alignment agreement below {args.min_agreement}')
        print(f'  excluded: {excluded}')

        rng = random.Random(args.seed)
        manifest = {
            'train_sentence_range': list(TRAIN_RANGE),
            'min_agreement': args.min_agreement,
            'seed': args.seed,
            'languages': {lg: {'agreement': round(d['agreement'], 4),
                               'coverage': round(d['coverage'], 4),
                               'segmenter': d['segmenter'],
                               'tier': d['tier'], 'script': d['script']}
                          for lg, d in loaded.items()},
            'excluded_languages': excluded,
            'rates': args.rates, 'modes': args.modes, 'counts': {},
        }

        for rate in args.rates:
            for mode in args.modes:
                rows = []
                per_lang = {}
                for lg, d in loaded.items():
                    lo, hi = TRAIN_RANGE
                    hi = min(hi, len(d['links']))
                    made = 0
                    for si in range(lo, hi):
                        if si >= len(d['links']):
                            break
                        res = switch_sentence(
                            d['src'][si], d['tgt'][si], d['links'][si],
                            rate, rng, mode,
                            drop_identity=not args.keep_identity)
                        if res is None:
                            continue
                        toks, sw = res
                        rows.append({'lang': lg, 'sentence_index': si,
                                     'rate': rate, 'mode': mode,
                                     'text': ' '.join(toks),
                                     'original': ' '.join(d['src'][si]),
                                     'n_switched': len(sw),
                                     'n_tokens': len(toks),
                                     'switches': sw})
                        made += 1
                    per_lang[lg] = made
                sub = os.path.join(OUT, f'rate{rate}_{mode}')
                os.makedirs(sub, exist_ok=True)
                with open(os.path.join(sub, 'corpus.jsonl'), 'w') as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                obs = (sum(r['n_switched'] for r in rows) /
                       max(sum(r['n_tokens'] for r in rows), 1))
                ident = sum(1 for r in rows for s in r['switches']
                            if s['source'].lower() == s['replacement'].lower())
                nsw = sum(r['n_switched'] for r in rows)
                manifest['counts'][f'rate{rate}_{mode}'] = {
                    'sentences': len(rows), 'per_language': per_lang,
                    'observed_switch_fraction_of_all_tokens': round(obs, 4),
                    'identity_substitutions': ident,
                    'identity_fraction': round(ident / max(nsw, 1), 4)}
                print(f'  [rate {rate} {mode:6s}] {len(rows):>6} sentences, '
                      f'{obs:.3f} of all tokens switched, '
                      f'identity substitutions {ident} '
                      f'({100*ident/max(nsw,1):.1f}%)', flush=True)

        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=1)
        ledger.close_run(rec, 'done', {'out': OUT,
                                       'languages': len(loaded)})
        print(f'[done] -> {OUT}')
    except Exception as e:
        ledger.close_run(rec, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
