#!/bin/bash
# export + convert the weight-bearing packed S=384 model with the op table
set -e; cd "$(dirname "$0")/.."
PYTHONPATH=pylib CALDEC_DENSE_ATTENTION=1 ${CALDEC_PY:-../../.venv/bin/python} packed/export_packed.py export --seq 384 --check 80 2>&1 | grep -v -i warn
RKNN_VERBOSE=1 rknn-venv/bin/python convert_generic.py packed/onnx/pk_s384.onnx packed/pk_s384.rknn > packed/conv_s384.log 2>&1
tail -3 packed/conv_s384.log
