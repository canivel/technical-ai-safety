# Source this before running uv commands:  source env.sh
# Code lives on P: (9p mount); venvs, models, adapters and activations live on ext4,
# because the 9p transport is slow and has wedged before (see ~/memlog_ext4.sh).
export SD_HOME="$HOME/work/safety-drift"
export SD_MODELS="$HOME/models"
export HF_HOME="$HOME/.cache/huggingface"
export UV_LINK_MODE=copy

# uv run / uv sync in the repo root -> training env; in serve/ -> vLLM env.
sd_train() { UV_PROJECT_ENVIRONMENT="$HOME/.venvs/sd-train" uv "$@"; }
sd_serve() { UV_PROJECT_ENVIRONMENT="$HOME/.venvs/sd-serve" uv --project "$(dirname "${BASH_SOURCE[0]}")/serve" "$@"; }
