#!/usr/bin/env bash
# Environment wrapper only. The caller owns process timeouts and resource limits.
set -euo pipefail
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/with_modern_vista_env.sh COMMAND [ARG ...]'
  echo 'Runs an operator-supplied command with read-only Git checks and offline model assets.'
  echo 'This is not an arbitrary-code sandbox or a process/resource supervisor.'
  exit 0
fi
command -v git >/dev/null || { echo 'Git is required to verify the pinned Vista source.' >&2; exit 1; }
export GIT_OPTIONAL_LOCKS=0
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
exec "$@"
