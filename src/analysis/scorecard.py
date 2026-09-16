#!/usr/bin/env python3
"""Battery scorecard: summarize cells.jsonl per model against the frozen
signature matrix's key rows. Emits numbers; judgment stays with the prereg.

Usage: python src/analysis/scorecard.py --model_tag Qwen3-1.7B-Base
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_tag', required=True)
    args = ap.parse_args()
    path = os.path.join(ROOT, 'results', 'battery', args.model_tag,
                        'cells.jsonl')
    rows = [json.loads(l) for l in open(path)]
    dip = rows[0]['layer']
    panel = {(r['injection'], r['magnitude'], r['rep']): r
             for r in rows if r['tier'] == 'panel' and r['layer'] == dip}
    base = panel[('I0-baseline', None, 0)]
    bb = base['budget']

    i10 = [r for r in rows if r['injection'] == 'I10'
           and r['tier'] == 'panel']
    floor_lfs = max(abs(r['lfs'] - base['lfs']) for r in i10)
    floor_mexa = max(abs(r['mexa_mean'] - base['mexa_mean']) for r in i10)

    print(f"=== {args.model_tag} @ dip L{dip}")
    print(f"baseline: LFS {base['lfs']:.4f} MEXA {base['mexa_mean']:.3f} "
          f"AaR {base['aar10_mean']:+.3f} effRank {base['eff_rank']:.1f} "
          f"probe {base.get('probe_acc', float('nan')):.3f}")
    print(f"I10 noise floor: dLFS {floor_lfs:.4f} dMEXA {floor_mexa:.4f}")
    print(f"baseline budget: " + ' '.join(
        f"{k[6:]}={v['mean']:+.3f}" for k, v in bb.items()))

    def show(inj, fields):
        for key in sorted(k for k in panel if k[0] == inj):
            r = panel[key]
            b = r['budget']
            parts = [f"{inj} m={key[1]}"]
            if 'dlfs' in fields:
                parts.append(f"dLFS {r['lfs'] - base['lfs']:+.4f}")
            if 'dmexa' in fields:
                parts.append(f"dMEXA {r['mexa_mean'] - base['mexa_mean']:+.3f}")
            if 'daar' in fields:
                parts.append(f"dAaR {r['aar10_mean'] - base['aar10_mean']:+.3f}")
            if 'rot' in fields:
                parts.append(f"rotSh {b['share_M3']['mean']:+.3f}"
                             f"(base {bb['share_M3']['mean']:+.3f})")
            if 'lin' in fields:
                parts.append(f"linSh {b['share_M4']['mean']:+.3f}")
            if 'nl' in fields:
                parts.append(f"nlSh {b['share_M5']['mean']:+.3f}")
            if 'off' in fields:
                parts.append(f"offSh {b['share_M1']['mean']:+.3f}"
                             f"(base {bb['share_M1']['mean']:+.3f})")
            if 'scale' in fields:
                parts.append(f"scSh {b['share_M2']['mean']:+.3f}")
            if 'cvp' in fields:
                parts.append(f"CVP {r['cvp']:.3f}")
            if 'rank' in fields:
                parts.append(f"dRank {r['eff_rank'] - base['eff_rank']:+.1f}")
            if 'probe' in fields:
                parts.append(f"probe {r.get('probe_acc', float('nan')):.3f}")
            if 'gap' in fields:
                parts.append(f"alGap {r.get('alignment_gap', 0):+.4f}")
            if 'cka' in fields:
                parts.append(f"ckaGap {r.get('cka_gap', 0):+.3f}")
            if 'theta' in fields and 'recovered_theta' in r:
                parts.append(f"recTheta {r['recovered_theta']['mean']:.3f}")
            if 'serr' in fields and 'scale_recovery_abs_err' in r:
                parts.append(f"sErr {r['scale_recovery_abs_err']:.4f}")
            print('  ' + '  '.join(parts))

    show('I1', ['dlfs', 'dmexa', 'off', 'probe'])
    show('I2', ['dlfs', 'dmexa', 'scale', 'serr'])
    show('I4', ['dlfs', 'dmexa', 'daar', 'rot', 'theta', 'gap'])
    show('I5', ['dlfs', 'rot', 'lin', 'scale'])
    show('I6', ['dlfs', 'dmexa', 'lin', 'nl', 'cka'])
    show('I7', ['dlfs', 'dmexa', 'nl', 'cka', 'rank'])
    show('I8', ['dlfs', 'dmexa', 'cvp', 'rank', 'scale', 'probe'])
    show('I9', ['dlfs', 'dmexa', 'daar', 'off', 'rot'])

    i3 = [r for r in rows if r['injection'] == 'I3']
    for r in i3:
        ref = base if r['layer'] == dip else None
        if ref:
            print(f"  I3 L{r['layer']} {r['tier']}: "
                  f"dLFS {r['lfs'] - ref['lfs']:+.5f} "
                  f"dMEXA {r['mexa_mean'] - ref['mexa_mean']:+.4f} "
                  f"(floor {floor_lfs:.4f}/{floor_mexa:.4f})")
        else:
            print(f"  I3 L{r['layer']} {r['tier']}: LFS {r['lfs']:.4f} "
                  f"MEXA {r['mexa_mean']:.3f} (cross-layer, no baseline row)")


if __name__ == '__main__':
    main()
