#!/usr/bin/env bash
set -euo pipefail

echo "=== ENV CHECK ==="
: "${DATABASE_URL:?DATABASE_URL is not set}"
echo "DATABASE_URL is set ✅"
echo "MODEL_VERSION=${MODEL_VERSION:-passing_v1}"

# always run relative to repo root, even if you call the script from elsewhere
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo "=== 1) INIT DB ==="
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$ROOT_DIR/scripts/init_db.sql"

echo ""
echo "=== 2) BUILD + LOAD MODEL DATA ==="
python "$ROOT_DIR/scripts/build_and_load.py"

echo ""
echo "=== 3) BUILD TRAIT DICTIONARY ==="
python "$ROOT_DIR/scripts/build_trait_dictionary.py"

echo ""
echo "✅ All done: DB initialised, data loaded, trait dictionary populated."
