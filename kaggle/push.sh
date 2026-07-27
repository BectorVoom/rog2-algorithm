#!/usr/bin/env bash
# Uploads the crate as a Kaggle dataset and pushes the T4 notebook.
#
#   KAGGLE_USER=boomvector ./kaggle/push.sh          # first time: creates the dataset
#   KAGGLE_USER=boomvector ./kaggle/push.sh --update # later: new dataset version
#
# The notebook needs GPU + internet, both set in kernel-metadata.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(dirname "$HERE")"
USER="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
SLUG="rog2-pf-src"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Ship sources only: target/ and dist/ are hundreds of MB and get rebuilt anyway.
# One tarball rather than a directory upload, because `kaggle datasets` only
# takes top-level files and its --dir-mode archives are not auto-extracted.
WORK="$STAGE/pack/rog2-algorithm"
mkdir -p "$WORK"
for item in Cargo.toml pyproject.toml README.md src python tests; do
  [ -e "$CRATE/$item" ] && cp -r "$CRATE/$item" "$WORK/"
done
tar -czf "$STAGE/rog2-src.tar.gz" -C "$STAGE/pack" rog2-algorithm
rm -rf "$STAGE/pack"

cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "rog2-pf-src",
  "id": "$USER/$SLUG",
  "licenses": [{ "name": "CC0-1.0" }]
}
JSON

echo "staged $(du -sh "$STAGE/rog2-src.tar.gz" | cut -f1) tarball"

if [ "${1:-}" = "--update" ]; then
  kaggle datasets version -p "$STAGE" -m "update crate sources"
else
  kaggle datasets create -p "$STAGE"
fi

python3 "$HERE/build_notebook.py"

# Attaching the competition data requires the pushing account to have accepted
# the competition rules; without it the notebook falls back to synthetic wells,
# which still exercises every code path. Set WITH_COMPETITION_DATA=1 once the
# rules are accepted.
COMP_SRC='"rogii-wellbore-geology-prediction"'
[ "${WITH_COMPETITION_DATA:-0}" = "1" ] || COMP_SRC=""

python3 - "$HERE/kernel-metadata.json" "$USER" "$COMP_SRC" <<'PYEOF'
import json, sys
path, user, comp = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.load(open(path))
meta["id"] = f"{user}/rog2-pf-cubecl-t4"
meta["dataset_sources"] = [f"{user}/rog2-pf-src"]
meta["competition_sources"] = [json.loads(comp)] if comp else []
json.dump(meta, open(path, "w"), indent=2)
open(path, "a").write("\n")
PYEOF

kaggle kernels push -p "$HERE"

echo "pushed: https://www.kaggle.com/code/$USER/rog2-pf-cubecl-t4"
