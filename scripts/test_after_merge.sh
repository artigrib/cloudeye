#!/usr/bin/env bash
# Full test suite after a merge, one summary line for the orchestrator.
# Usage: scripts/test_after_merge.sh [extra pytest args]
# Prints:  TESTS passed=<n> failed=<n> skipped=<n> errors=<n> | FAILED: <test ids>
# Exit code: pytest's exit code (0 = all green).
set -u
cd "$(dirname "$0")/.."
report="$(mktemp -t test_after_merge.XXXXXX)"
uv run pytest tests -q -p no:cacheprovider -rfE "$@" >"$report" 2>&1
rc=$?
passed=$(grep -Eo '[0-9]+ passed' "$report" | tail -1 | grep -Eo '[0-9]+' || echo 0)
failed=$(grep -Eo '[0-9]+ failed' "$report" | tail -1 | grep -Eo '[0-9]+' || echo 0)
skipped=$(grep -Eo '[0-9]+ skipped' "$report" | tail -1 | grep -Eo '[0-9]+' || echo 0)
errors=$(grep -Eo '[0-9]+ error(s)?' "$report" | tail -1 | grep -Eo '[0-9]+' || echo 0)
names=$(grep -E '^(FAILED|ERROR) ' "$report" | sed -E 's/^(FAILED|ERROR) //; s/ - .*$//' | paste -sd ',' -)
line="TESTS passed=${passed:-0} failed=${failed:-0} skipped=${skipped:-0} errors=${errors:-0}"
if [ -n "$names" ]; then line="$line | FAILED: $names"; fi
if [ $rc -ne 0 ] && [ "${failed:-0}" = "0" ] && [ "${errors:-0}" = "0" ]; then
  line="$line | pytest exit $rc (see $report)"
else
  rm -f "$report"
fi
echo "$line"
exit $rc
