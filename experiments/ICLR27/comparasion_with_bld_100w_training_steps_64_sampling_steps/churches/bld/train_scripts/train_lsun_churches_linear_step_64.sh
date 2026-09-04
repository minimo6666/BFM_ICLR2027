#!/usr/bin/env bash
# Compatibility wrapper for the canonical 64-step BLD launcher.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_lsun_churches_64_steps.sh" "$@"
