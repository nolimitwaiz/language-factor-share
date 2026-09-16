#!/usr/bin/env python3
"""The determinism gate.

Compares every reported quantity between two runs of the same
configuration and exits non-zero if any differs by more than the tolerance.

    python experiments/compare_runs.py A.json B.json [--tol 1e-6]

Also reports whether the provenance blocks agree, since two runs that agree
numerically but differ in model revision or dependency versions are not
evidence of determinism.
"""
import argparse
import json
import sys


def walk(node, path=''):
    """Yield (dotted_path, float) for every numeric leaf."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f'{path}.{k}' if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f'{path}[{i}]')
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        yield path, float(node)


# Quantities that legitimately differ between runs and are not part of the gate.
EXEMPT = ('wallclock_s', 'provenance.slurm_job', 't_start')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--tol', type=float, default=None)
    args = ap.parse_args()

    A = json.load(open(args.a))
    B = json.load(open(args.b))
    tol = args.tol
    if tol is None:
        tol = (A.get('provenance', {}).get('config', {})
                .get('determinism_gate', {}).get('tolerance', 1e-6))

    na = {k: v for k, v in walk(A) if not any(e in k for e in EXEMPT)}
    nb = {k: v for k, v in walk(B) if not any(e in k for e in EXEMPT)}

    only_a, only_b = set(na) - set(nb), set(nb) - set(na)
    if only_a or only_b:
        print(f'STRUCTURE MISMATCH: {len(only_a)} keys only in A, '
              f'{len(only_b)} only in B')
        for k in list(only_a)[:5] + list(only_b)[:5]:
            print(f'  {k}')
        sys.exit(2)

    diffs = []
    for k in sorted(na):
        d = abs(na[k] - nb[k])
        if d > tol:
            diffs.append((k, na[k], nb[k], d))

    # provenance comparison, reported but not gated on
    pa = A.get('provenance', {})
    pb = B.get('provenance', {})
    prov_keys = ['config_sha256_16', 'src_hash', 'data_digests', 'versions']
    prov_diff = [k for k in prov_keys if pa.get(k) != pb.get(k)]

    print(f'compared {len(na)} numeric quantities at tolerance {tol:g}')
    if prov_diff:
        print(f'  PROVENANCE DIFFERS in: {", ".join(prov_diff)} '
              f'(runs are not directly comparable)')
    else:
        print('  provenance identical (config, source tree, data, versions)')

    if not diffs:
        print(f'DETERMINISM GATE: PASS. Maximum absolute difference '
              f'{max((abs(na[k] - nb[k]) for k in na), default=0):.3e}')
        sys.exit(0)

    print(f'DETERMINISM GATE: FAIL. {len(diffs)} quantities exceed tolerance.')
    for k, x, y, d in sorted(diffs, key=lambda t: -t[3])[:15]:
        print(f'  {k}: {x!r} vs {y!r}  (delta {d:.3e})')
    sys.exit(1)


if __name__ == '__main__':
    main()
