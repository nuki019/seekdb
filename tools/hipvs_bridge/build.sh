#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM_ROOT/bin/hipcc}
OUTPUT=${1:-$SCRIPT_DIR/libseekdb_cuvs_bridge.so}

[[ -x "$HIPCC" ]] || { echo "hipcc is not executable: $HIPCC" >&2; exit 2; }
[[ -f "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || {
  echo "hipVS headers are missing under $HIPVS_ROOT/include" >&2
  exit 2
}
[[ -f "$HIPVS_ROOT/lib/libcuvs_c.so" ]] || {
  echo "libcuvs_c.so is missing under $HIPVS_ROOT/lib" >&2
  exit 2
}

mkdir -p "$(dirname "$OUTPUT")"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
# GPU kernels target the Radeon PRO W7900 family explicitly (the build host may have no GPU).
GPU_ARCH=${GPU_ARCH:-gfx1100}

"$HIPCC" \
  -c \
  -fPIC \
  -O2 \
  -D__HIP_PLATFORM_AMD__ \
  -I"$HIPVS_ROOT/include" \
  -I"$ROCM_ROOT/include" \
  "$SCRIPT_DIR/seekdb_cuvs_bridge.c" \
  -o "$TMP_DIR/seekdb_cuvs_bridge.o"

"$HIPCC" \
  -c \
  -fPIC \
  -O3 \
  --offload-arch="$GPU_ARCH" \
  -D__HIP_PLATFORM_AMD__ \
  -I"$HIPVS_ROOT/include" \
  -I"$ROCM_ROOT/include" \
  -x hip "$SCRIPT_DIR/seekdb_cuvs_graph.hip" \
  -o "$TMP_DIR/seekdb_cuvs_graph.o"

"$HIPCC" \
  -shared \
  -fPIC \
  --offload-arch="$GPU_ARCH" \
  "$TMP_DIR/seekdb_cuvs_bridge.o" \
  "$TMP_DIR/seekdb_cuvs_graph.o" \
  -L"$HIPVS_ROOT/lib" \
  -Wl,-soname,libseekdb_cuvs_bridge.so \
  -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" \
  -L"$ROCM_ROOT/lib" \
  -lcuvs_c \
  -lrocblas \
  -o "$OUTPUT"

echo "built $OUTPUT"
