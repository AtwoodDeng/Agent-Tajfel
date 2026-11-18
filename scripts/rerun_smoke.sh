#!/usr/bin/env bash
# Re-run the 3-arm scripted smoke test after the model fixes, archiving the pre-fix runs.
#
#   bash scripts/rerun_smoke.sh
#
# The pre-fix runs in runs/ were produced by a model with three defects (degenerate MIF
# matrix, a sign-inverted favour metric, and a scripted stub that could not tell c1 from c3),
# so they are moved aside rather than mixed with the new data.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p runs/_archive
for d in c1 c2 c3 c1_openai; do
  if [ -d "runs/$d" ]; then
    mv "runs/$d" "runs/_archive/${d}_pre-fix_${STAMP}"
    echo "archived runs/$d -> runs/_archive/${d}_pre-fix_${STAMP}"
  fi
done

echo
echo "== selftest =="
python scripts/selftest_tajfel.py

echo
echo "== 3-arm scripted smoke test (10 agents x 3 steps) =="
python scripts/run_tajfel.py --all --n-agents 10 --n-steps 3

echo
echo "== last-step metrics per arm =="
for d in c1 c2 c3; do
  echo "-- $d"
  python - "$d" <<'PY'
import json, sys
from pathlib import Path
d = sys.argv[1]
s = json.loads(Path(f"runs/{d}/export/summary.json").read_text(encoding="utf-8"))
for k, v in s.get("last_step", {}).items():
    print(f"   {k:22} {v}")
PY
done

echo
echo "== defect regressions =="
fail=0
if cmp -s runs/c1/export/metrics.csv runs/c3/export/metrics.csv; then
  echo "   FAIL c1 and c3 metrics are identical (the rarity condition is not reaching the model)"
  fail=1
else
  echo "   OK   c1 and c3 differ"
fi
if cmp -s runs/c1/export/allocations.csv runs/c2/export/allocations.csv; then
  echo "   FAIL c1 and c2 allocations are identical (label visibility is not reaching the model)"
  fail=1
else
  echo "   OK   c1 and c2 differ"
fi
if head -1 runs/c1/export/allocations.csv | grep -q ',own_group,'; then
  echo "   OK   allocations.csv carries own_group for per-allocator stratification"
else
  echo "   FAIL allocations.csv is missing the own_group column"
  fail=1
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "SMOKE TEST FAILED"
  exit 1
fi
echo "SMOKE TEST PASSED -- runs/c1, runs/c2, runs/c3 are valid pipeline data"
echo "(scripted mode is a pipeline probe: its numbers are NOT evidence about the effect)"
