#!/usr/bin/env bash
# Build the front-end into ui/dist, which the Rust crate embeds.
set -euo pipefail
cd "$(dirname "$0")"
npm install --no-audit --no-fund
npm run build
echo "Built $(find dist -type f | wc -l) files into ui/dist"
