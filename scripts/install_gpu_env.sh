#!/usr/bin/env bash
# set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-crestfc}"
cd "$REPO_ROOT"
conda create -y -n "$ENV_NAME" python=3.10.15
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" python -c 'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.cuda.is_available(), "PyTorch cannot access the GPU"; assert torch.version.cuda.startswith("12."); print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
printf 'Installed. Activate with: source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate %s\n' "$ENV_NAME"
printf 'Then run: PYTHON_BIN="$(command -v python)" scripts/run_experiments.sh e0 --run-id pilot\n'
