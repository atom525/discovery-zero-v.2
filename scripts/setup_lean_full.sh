#!/usr/bin/env bash
# Discovery-Zero-v2 Lean 一键配置（自包含版本）
# 用法: ./scripts/setup_lean_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEAN_WORKSPACE="$PROJECT_ROOT/lean_workspace"
LEAN_VERSION="v4.29.0-rc4"

export PATH="$HOME/.elan/bin:${PATH:-}"

echo "=== Discovery-Zero-v2 Lean 一键配置 ==="

echo ">>> [1/6] 清理旧产物"
pkill -9 -f "lake " 2>/dev/null || true
rm -f ~/.elan/toolchains/*.lock 2>/dev/null || true
rm -f ~/.elan/tmp/*.lock 2>/dev/null || true
rm -rf "$LEAN_WORKSPACE/.lake" 2>/dev/null || true
find "$LEAN_WORKSPACE" -name "*.olean" -delete 2>/dev/null || true

echo ">>> [2/6] 安装 elan（如未安装）"
if ! command -v elan >/dev/null 2>&1; then
  if ! curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none; then
    ARCH="x86_64-unknown-linux-gnu"
    [[ "$(uname -m)" == "aarch64" ]] && ARCH="aarch64-unknown-linux-gnu"
    ELAN_VER="v4.0.0"
    MIRROR_URL="https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/elan/elan/releases/download/${ELAN_VER}/elan-${ARCH}.tar.gz"
    TMP_DIR="/tmp/elan-install-$$"
    mkdir -p "$TMP_DIR"
    cd "$TMP_DIR"
    curl -L -o elan.tar.gz "$MIRROR_URL" || curl -L -o elan.tar.gz "https://github.com/leanprover/elan/releases/download/${ELAN_VER}/elan-${ARCH}.tar.gz"
    tar xf elan.tar.gz
    chmod +x elan-init
    ./elan-init -y --default-toolchain none
    cd /
    rm -rf "$TMP_DIR"
  fi
fi
[[ -f "$HOME/.elan/env" ]] && source "$HOME/.elan/env"
export PATH="$HOME/.elan/bin:${PATH:-}"

echo ">>> [3/6] 安装/切换 Lean toolchain: $LEAN_VERSION"
elan default leanprover/lean4:${LEAN_VERSION} 2>/dev/null || elan default leanprover/lean4:stable

echo ">>> [4/6] 初始化 lean_workspace"
mkdir -p "$LEAN_WORKSPACE/Discovery"
if [[ ! -f "$LEAN_WORKSPACE/lakefile.toml" ]]; then
  (cd "$PROJECT_ROOT" && pip install -e . -q 2>/dev/null || true)
  if command -v dz >/dev/null 2>&1; then
    dz lean init --path "$LEAN_WORKSPACE"
  else
    (cd "$PROJECT_ROOT" && LEAN_WORKSPACE="$LEAN_WORKSPACE" PYTHONPATH=src python - <<'PY'
import os
from pathlib import Path
from discovery_zero.tools.lean import init_workspace
init_workspace(Path(os.environ["LEAN_WORKSPACE"]))
PY
    )
  fi
fi

echo ">>> [5/6] lake update（多镜像重试）"
MIRRORS=(
  "https://mirror.ghproxy.com/https://github.com/leanprover-community/mathlib4.git"
  "https://ghproxy.net/https://github.com/leanprover-community/mathlib4.git"
  "https://github.com/leanprover-community/mathlib4.git"
)
LAKE_OK=false
for MIRROR in "${MIRRORS[@]}"; do
  sed -i "s|git = \"[^\"]*\"|git = \"$MIRROR\"|" "$LEAN_WORKSPACE/lakefile.toml" 2>/dev/null || true
  if (cd "$LEAN_WORKSPACE" && lake update 2>/dev/null); then
    LAKE_OK=true
    break
  fi
done
if [[ "$LAKE_OK" != true ]]; then
  echo "[失败] lake update 失败，请检查网络后重试。"
  exit 1
fi

echo ">>> [6/6] cache get + lake build"
(cd "$LEAN_WORKSPACE" && lake exe cache get 2>/dev/null || true)
(cd "$LEAN_WORKSPACE" && lake build)

echo ""
echo "=== 完成 ==="
echo "验证命令: cd $LEAN_WORKSPACE && lake build"
