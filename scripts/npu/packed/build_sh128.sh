#!/bin/bash
# experiment: S=256 with head rows capped at 128 (full weights, standalone), its own cases dir
set -e; cd "$(dirname "$0")/.."
export SH_MAX=128
cp -a packed/tables_s256 packed/tables_s256.keep
PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1 ${CALDEC_PY:-../../.venv/bin/python} packed/export_packed.py export --seq 256 --check 40 2>&1 | grep -E "^S=|wrote"
mv packed/onnx/pk_s256.onnx packed/onnx/pk_sh128_s256.onnx; mv packed/tables_s256 packed/tables_sh128_s256; mv packed/tables_s256.keep packed/tables_s256
rknn-venv/bin/python convert_generic.py packed/onnx/pk_sh128_s256.onnx packed/pk_sh128_s256.rknn > packed/conv_sh128.log 2>&1
tail -1 packed/conv_sh128.log; echo DONE
