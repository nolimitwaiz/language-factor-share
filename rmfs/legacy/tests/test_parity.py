"""Golden-file parity gate: LFS/MEXA recomputed from Phase-0 dumps must match
the prior grid numbers. Skips until dumps exist; tolerances are [DECIDE]
prereg fields — the values here are the spec's suggested defaults and the
hardware caveat applies (dumps computed on different hardware than the
original grid may differ beyond quantization; judge at STOP-1).
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPS = os.path.join(ROOT, 'results', 'dumps')

TOL_LFS = 0.005    # [DECIDE] prereg field
TOL_MEXA = 0.01    # [DECIDE] prereg field


def _dumped_models():
    if not os.path.isdir(DUMPS):
        return []
    out = []
    for tag in sorted(os.listdir(DUMPS)):
        meta = os.path.join(DUMPS, tag, 'meta.json')
        grid = os.path.join(ROOT, 'results', 'grid', tag, 'metrics.json')
        if os.path.exists(meta) and os.path.exists(grid):
            m = json.load(open(meta))
            # parity only meaningful at full grid config (n=300, all langs)
            if m.get('n_sents') == 300 and len(m.get('langs', [])) > 100:
                out.append(tag)
    return out


@pytest.mark.parametrize('tag', _dumped_models() or ['__no_dumps__'])
def test_parity_against_grid(tag):
    if tag == '__no_dumps__':
        pytest.skip('no Phase-0 dumps yet — run dump_embeddings first')
    from src.campaign.dump_embeddings import parity_against_grid
    rows = parity_against_grid(tag)
    assert rows, f'no layers compared for {tag}'
    for row in rows:
        assert row['d_lfs'] <= TOL_LFS, (tag, row)
        if row['d_mexa_mean'] is not None:
            assert row['d_mexa_mean'] <= TOL_MEXA, (tag, row)
