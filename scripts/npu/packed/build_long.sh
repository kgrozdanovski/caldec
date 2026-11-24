#!/bin/bash
# long-state buckets for the pk2 set (weightless, share pk2_s384.rknn's weights): S=512, 640. Low priority, 4 cores.
set -e; cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="env PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1 ACT=1 ${CALDEC_PY:-../../.venv/bin/python}"
for S in 512 640; do
  $PY packed/export_packed.py export --seq $S --check 10 2>&1 | grep -E "^S=|wrote"
  RKNN_VERBOSE=1 rknn-venv/bin/python convert_generic.py packed/onnx/pk2_s$S.onnx packed/pk2_s${S}_now.rknn remove_weight=True > packed/conv2_s$S.log 2>&1
  echo "S=$S $(ls -la packed/pk2_s${S}_now.rknn)"; rm -f packed/onnx/pk2_s$S.onnx*
done
echo ALL_DONE
