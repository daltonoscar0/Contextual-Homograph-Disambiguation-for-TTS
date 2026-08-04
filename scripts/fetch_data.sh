#!/usr/bin/env bash
# Fetch the Wikipedia Homograph Data set and CMUdict into data/vendor/.
# Both are skipped if already present, so this is cheap to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/data/vendor"
mkdir -p "$VENDOR/cmudict"

WHD="$VENDOR/WikipediaHomographData"
if [ -d "$WHD/data" ]; then
  echo "homograph data already present, skipping"
else
  echo "fetching WikipediaHomographData"
  if ! git clone --depth 1 -q \
      https://github.com/google-research-datasets/WikipediaHomographData "$WHD"; then
    echo "clone failed, falling back to tarball"
    rm -rf "$WHD"
    curl -fsSL -o "$VENDOR/whd.tar.gz" \
      https://github.com/google-research-datasets/WikipediaHomographData/archive/refs/heads/master.tar.gz
    tar -xzf "$VENDOR/whd.tar.gz" -C "$VENDOR"
    mv "$VENDOR/WikipediaHomographData-master" "$WHD"
    rm -f "$VENDOR/whd.tar.gz"
  fi
fi

CMU_BASE="https://raw.githubusercontent.com/cmusphinx/cmudict/master"
for f in cmudict.dict cmudict.phones; do
  if [ -s "$VENDOR/cmudict/$f" ]; then
    echo "cmudict/$f already present, skipping"
  else
    echo "fetching cmudict/$f"
    curl -fsSL -o "$VENDOR/cmudict/$f" "$CMU_BASE/$f"
  fi
done

echo
echo "train files: $(ls "$WHD/data/train" | wc -l | tr -d ' ')"
echo "eval files:  $(ls "$WHD/data/eval" | wc -l | tr -d ' ')"
echo "cmudict:     $(wc -l < "$VENDOR/cmudict/cmudict.dict" | tr -d ' ') entries"
