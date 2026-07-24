#!/usr/bin/env bash
# Syncs the canonical dataset from the upstream drishti_pipeline output into
# this repo's dataset/ dir. Run this before train_model.py whenever the
# upstream pipeline has regenerated the dataset, so training can't silently
# drift from a stale local copy.
#
# Usage: ./sync_dataset.sh [path-to-drishti_dataset-dir]
# Default source is this dev machine's pipeline checkout; override with $1
# or DRISHTI_DATASET_DIR since that path is machine-local, not portable.

set -euo pipefail

SRC="${1:-${DRISHTI_DATASET_DIR:-/media/sandesh/extra-ssd/dataset/dataset-make/drishti_dataset}}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dataset"

if [ ! -f "$SRC/canonical_dataset.csv" ]; then
    echo "ERROR: $SRC/canonical_dataset.csv not found. Pass the drishti_dataset dir as \$1." >&2
    exit 1
fi

for f in canonical_dataset.csv canonical_dataset_prenoise.csv metadata.json noise_log.json; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$DEST/$f"
        echo "synced $f"
    else
        echo "WARNING: $SRC/$f not found, skipped" >&2
    fi
done

python3 - "$DEST/canonical_dataset.csv" <<'EOF'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
print(f"\nrows: {len(df)}")
print("tier distribution:")
print(df["tier"].value_counts(normalize=True).sort_index())
print(f"glucose notna: {df['glucose'].notna().mean():.1%}" if "glucose" in df.columns else "WARNING: no glucose column")
EOF
