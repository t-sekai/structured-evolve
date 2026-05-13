#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" == "--target" ]]; then
    echo "run_all.sh sets --target internally; pass shape/output arguments only." >&2
    exit 2
  fi
done

python scripts/run_baseline.py --target llvm "$@"

if python -c 'import tvm; raise SystemExit(0 if tvm.cuda(0).exist else 1)'; then
  python scripts/run_baseline.py --target cuda "$@"
else
  echo "Skipping CUDA baseline: tvm.cuda(0) is unavailable."
fi
