#!/bin/bash
# Stage the CalDec NPU runner on a device (run on the PC). Usage: CALDEC_CKPT_DIR=<local checkpoint dir> runner/stage.sh user@board
# Device layout: ~/caldec-npu/{caldec_npu.py, caldec_npu_server(.c), models.txt, tokenizer*.json, rl_agent_config.json,
# pk_s*.rknn -> ~/npu-bench/packed/, tables_s* -> ~/npu-bench/packed/, venv/ (numpy, tokenizers)}
set -e; H=$1; cd "$(dirname "$0")/.."
O="${SSH_OPTS:-}"
CK=${CALDEC_CKPT_DIR:?set CALDEC_CKPT_DIR to the local checkpoint directory}
ssh $O $H 'mkdir -p ~/caldec-npu'
rsync -a -e "ssh $O" runner/caldec_npu.py runner/caldec_npu_server.c $CK/tokenizer/tokenizer.json $CK/tokenizer/tokenizer_config.json $CK/rl_agent_config.json $H:caldec-npu/
ssh $O $H 'set -e; cd ~/caldec-npu
  for f in ~/npu-bench/packed/pk_s*.rknn; do case $f in *sh128*) continue;; esac; ln -sf $f .; done
  for d in ~/npu-bench/packed/tables_s[0-9]*; do ln -sfn $d .; done
  { echo 8; echo "384 pk_s384.rknn tables_s384 0"; for s in 96 128 160 192 224 256 320; do echo "$s pk_s${s}_now.rknn tables_s$s 1"; done; } > models.txt
  ln -sf ~/npu-bench/rknn_api.h ~/npu-bench/librknnrt.so .
  gcc -O2 -o caldec_npu_server caldec_npu_server.c -I. -L. -lrknnrt -Wl,-rpath,"\$ORIGIN"
  [ -x venv/bin/python ] || { python3 -m venv venv && venv/bin/pip install -q numpy tokenizers; }
  venv/bin/python -c "import numpy, tokenizers; print(\"numpy\", numpy.__version__, \"tokenizers\", tokenizers.__version__)"'
