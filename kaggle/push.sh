#!/usr/bin/env bash
# Uploads the crate as a Kaggle dataset and pushes the T4 notebooks.
#
#   KAGGLE_USER=boomvector ./kaggle/push.sh                  # first run: creates the dataset
#   KAGGLE_USER=boomvector ./kaggle/push.sh --update         # later runs: new dataset version
#   KAGGLE_USER=boomvector ./kaggle/push.sh --notebook beam  # only the beam-search notebook
#
# `--notebook` takes pf, beam or all (default all).
# The notebooks need GPU + internet, both set in their kernel-metadata files.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(dirname "$HERE")"
USER="${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
SLUG="rog2-pf-src"

UPDATE=0
NOTEBOOK="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --update) UPDATE=1; shift ;;
    --notebook) NOTEBOOK="${2:?--notebook needs pf|beam|all}"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$NOTEBOOK" in
  pf|beam|all) ;;
  *) echo "--notebook must be pf, beam or all" >&2; exit 2 ;;
esac

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

if [ "$UPDATE" = "1" ]; then
  kaggle datasets version -p "$STAGE" -m "update crate sources"
else
  kaggle datasets create -p "$STAGE"
fi

python3 "$HERE/build_notebook.py"
python3 "$HERE/build_beam_notebook.py"

# Attaching the competition data requires the pushing account to have accepted
# the competition rules; without it the notebooks fall back to synthetic wells,
# which still exercises every code path. Set WITH_COMPETITION_DATA=1 once the
# rules are accepted.
COMP_SRC='"rogii-wellbore-geology-prediction"'
[ "${WITH_COMPETITION_DATA:-0}" = "1" ] || COMP_SRC=""

# `kaggle kernels push` takes a *directory* holding exactly one
# kernel-metadata.json, so each notebook goes out from its own staging dir.
push_one() {
  local meta="$1" slug="$2"
  local dir="$STAGE/push-$slug"
  mkdir -p "$dir"

  python3 - "$HERE/$meta" "$dir/kernel-metadata.json" "$USER" "$slug" "$COMP_SRC" <<'PYEOF'
import json, sys
src, dst, user, slug, comp = sys.argv[1:6]
meta = json.load(open(src))
meta["id"] = f"{user}/{slug}"
meta["dataset_sources"] = [f"{user}/rog2-pf-src"]
meta["competition_sources"] = [json.loads(comp)] if comp else []
for path in (dst, src):          # keep the checked-in copy in step with the push
    json.dump(meta, open(path, "w"), indent=2)
    open(path, "a").write("\n")
PYEOF

  cp "$HERE/$slug.ipynb" "$dir/"
  kaggle kernels push -p "$dir"
  echo "pushed: https://www.kaggle.com/code/$USER/$slug"
}

if [ "$NOTEBOOK" = "pf" ] || [ "$NOTEBOOK" = "all" ]; then
  push_one kernel-metadata.json rog2-pf-cubecl-t4
fi
if [ "$NOTEBOOK" = "beam" ] || [ "$NOTEBOOK" = "all" ]; then
  push_one kernel-metadata-beam.json rog2-beam-cubecl-t4
fi
