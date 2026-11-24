#!/bin/bash
# weightless (remove_weight) packed buckets that share pk_s384.rknn's weights at run time
set -e; cd "$(dirname "$0")/.."
for S in 256 320 224 192 160 128 96; do
  PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1 ${CALDEC_PY:-../../.venv/bin/python} packed/export_packed.py export --seq $S --check 60 2>&1 | grep -E "^S=|wrote"
  RKNN_VERBOSE=1 rknn-venv/bin/python convert_generic.py packed/onnx/pk_s$S.onnx packed/pk_s${S}_now.rknn remove_weight=True > packed/conv_s$S.log 2>&1
  echo "S=$S cpu_ops=$(awk '/OpType .*Target/{f=1;next} f && /CPU/' packed/conv_s$S.log | grep -vc InputOperator) $(tail -1 packed/conv_s$S.log)"
  rm -f packed/onnx/pk_s$S.onnx
done
echo ALL_DONE
