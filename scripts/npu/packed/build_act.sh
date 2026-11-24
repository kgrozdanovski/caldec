#!/bin/bash
# packed set with the act-head rows as a second output (ACT=1): pk2_s384.rknn (weights) + weightless pk2_s*_now.rknn
set -e; cd "$(dirname "$0")/.."
PY="env PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1 ACT=1 ${CALDEC_PY:-../../.venv/bin/python}"
$PY packed/export_packed.py export --seq 384 --check 40 2>&1 | grep -E "^S=|wrote"
RKNN_VERBOSE=1 rknn-venv/bin/python convert_generic.py packed/onnx/pk2_s384.onnx packed/pk2_s384.rknn > packed/conv2_s384.log 2>&1
echo "S=384 $(tail -1 packed/conv2_s384.log)"; rm -f packed/onnx/pk2_s384.onnx*
for S in 256 320 224 192 160 128 96; do
  $PY packed/export_packed.py export --seq $S --check 20 2>&1 | grep -E "^S=|wrote"
  RKNN_VERBOSE=1 rknn-venv/bin/python convert_generic.py packed/onnx/pk2_s$S.onnx packed/pk2_s${S}_now.rknn remove_weight=True > packed/conv2_s$S.log 2>&1
  echo "S=$S $(tail -1 packed/conv2_s$S.log)"; rm -f packed/onnx/pk2_s$S.onnx*
done
echo ALL_DONE
