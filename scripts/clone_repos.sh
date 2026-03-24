#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
cd external

git clone https://github.com/pengzhangzhi/Open-dLLM.git || (cd Open-dLLM && git pull)
git clone https://github.com/alisawuffles/ambient.git || (cd ambient && git pull)

echo "Repos cloned into ./external/"
