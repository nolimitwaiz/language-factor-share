"""Run ledger: every run appends one JSON line to ledger.jsonl at repo root.

Records run_id, config hash, seed, phase, inputs, outputs, wallclock, status —
including dead/killed runs (call `open_run` early; `close_run` in finally).
The repo is not under git, so we hash the source tree instead of a commit.
"""
import fcntl
import hashlib
import json
import os
import socket
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = os.path.join(ROOT, 'ledger.jsonl')


def _src_hash():
    """Hash of all .py files under src/ + code/ — the no-git commit stand-in."""
    h = hashlib.sha256()
    for base in ('src', 'code'):
        d = os.path.join(ROOT, base)
        if not os.path.isdir(d):
            continue
        for dirpath, _, files in sorted(os.walk(d)):
            for fn in sorted(files):
                if fn.endswith('.py'):
                    with open(os.path.join(dirpath, fn), 'rb') as f:
                        h.update(f.read())
    return h.hexdigest()[:16]


def config_hash(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def open_run(phase, config, seed=None):
    rec = {
        'run_id': uuid.uuid4().hex[:12],
        'phase': phase,
        'src_hash': _src_hash(),
        'config': config,
        'config_hash': config_hash(config),
        'seed': seed,
        'host': socket.gethostname(),
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
        't_start': time.time(),
        'status': 'running',
    }
    _append({**rec, 'event': 'start'})
    return rec


def close_run(rec, status, outputs=None):
    rec = dict(rec)
    rec.update(status=status, outputs=outputs or {},
               wallclock_s=round(time.time() - rec['t_start'], 1))
    _append({**rec, 'event': 'end'})
    return rec


def _append(rec):
    """Append one JSON line under an exclusive advisory lock.

    Concurrent SLURM array tasks share this file. Without the lock a
    buffered write can be split across syscalls and interleave with another
    task's write, producing a torn line; one such line was found in the
    campaign ledger (2026-07-24 audit) and is recorded in
    docs/DISCREPANCY_LEDGER.md. Lines written before this fix are otherwise
    intact."""
    line = json.dumps(rec, default=str) + '\n'
    with open(LEDGER, 'a') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
