import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

# Apple Accelerate BLAS raises spurious FPE warnings in matmul on
# verified-finite input (numpy/arm64 quirk — documented in cluster_grid.py).
# Tests assert finiteness explicitly where it matters; silence the noise.
warnings.filterwarnings(
    'ignore', message='.*encountered in matmul', category=RuntimeWarning)
