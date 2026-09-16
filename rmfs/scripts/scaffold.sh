#!/usr/bin/env bash
# Scaffold the RMFS research repository structure. Run from the repo root.
set -euo pipefail

mkdir -p configs/experiments \
         prereg/signatures prereg/addenda \
         src/rmfs/data src/rmfs/extraction src/rmfs/components \
         src/rmfs/metrics src/rmfs/validity src/rmfs/battery src/rmfs/utils \
         scripts/sbatch \
         tests \
         data/raw data/embeddings data/external \
         legacy \
         results/runs results/tables results/figures \
         ledgers \
         paper/sections paper/appendix

touch src/rmfs/__init__.py \
      src/rmfs/data/__init__.py src/rmfs/extraction/__init__.py \
      src/rmfs/components/__init__.py src/rmfs/metrics/__init__.py \
      src/rmfs/validity/__init__.py src/rmfs/battery/__init__.py \
      src/rmfs/utils/__init__.py

for f in ERROR_LEDGER DECISIONS RUNS INVENTORY; do
  [ -f "ledgers/${f}.md" ] || printf '# %s\n\n' "$f" > "ledgers/${f}.md"
done

if [ ! -f .gitignore ]; then
cat > .gitignore <<'GITEOF'
data/
results/
!results/runs/**/manifest.json
__pycache__/
*.pyc
.venv/
.ruff_cache/
.pytest_cache/
GITEOF
fi

echo "Scaffold complete."
