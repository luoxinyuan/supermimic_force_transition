#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source env.sh
exec .venv/bin/python experiments/full_tracking/download.py
