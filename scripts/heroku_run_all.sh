#!/usr/bin/env bash
set -euo pipefail

echo "=== ENV CHECK ==="
: "${DATABASE_URL:?DATABASE_URL is not set}"
echo "DATABASE_URL is set ✅"
echo "MODEL_VERSION=${MODEL_VERSION:-passing_v1}"

echo ""
echo "=== 1) INIT DB ==="
run scripts/init_db.sql

echo ""
echo "=== 2) BUILD + LOAD MODEL DATA ==="
python scripts/build_and_load.py

echo ""
echo "=== 3) BUILD TRAIT DICTIONARY ==="
python scripts/build_trait_dictionary.py

echo ""
echo "✅ All done: DB initialised, data loaded, trait dictionary populated."
