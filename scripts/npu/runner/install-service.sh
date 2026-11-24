#!/bin/bash
# Install the CalDec NPU service on a board (run on the PC). Usage: runner/install-service.sh <user@board> [set]
#   set = pk2 (default: packed set with the act-head output) or pk (option logits only)
# Models: NPU_MODELS = a directory with <set>_s*.rknn, tables_s*/, act_head.npz, tokenizer/ and rl_agent_config.json
#   (the Hugging Face repo kgrozdanovski/caldec-ultra-150m-rknn-rk3566 as downloaded). Default: the build outputs
#   here (packed/, runner/act_head.npz) with CALDEC_CKPT_DIR for the tokenizer and config.
# Node layout: /opt/caldec-npu (root-owned, read-only for the service user "caldec"):
#   caldec_npu.py, caldec_npu_service.py, caldec_npu_server (built here), librknnrt.so, models.txt, <set>_s*.rknn,
#   tables_s*/, tokenizer*.json, rl_agent_config.json, act_head.npz, venv/ (numpy, tokenizers, pinned)
# Units: caldec-npu.socket (/run/caldec-npu/caldec.sock, caldec:caldec-clients 0660) -> caldec-npu.service (idle exit 600 s).
# Client: /usr/local/bin/caldec-npu. Users in group caldec-clients may connect.
# SSH_OPTS: extra ssh options (key, known_hosts).
set -euo pipefail; H=$1; SET=${2:-pk2}; cd "$(dirname "$0")/.."
O="${SSH_OPTS:-}"
if [ -n "${NPU_MODELS:-}" ]; then P=$NPU_MODELS; CK=${CALDEC_CKPT_DIR:-$NPU_MODELS}; AH=$NPU_MODELS/act_head.npz
else P=packed; CK=${CALDEC_CKPT_DIR:?set NPU_MODELS, or CALDEC_CKPT_DIR to the local checkpoint directory}; AH=runner/act_head.npz; fi
RT=${RKNN_RT:?set RKNN_RT to a directory with librknnrt.so and rknn_api.h (rknn-toolkit2 2.5.0 runtime, aarch64)}
# Staging: a private (0700), unpredictable directory made by the deploying user; the install script lives in it and
# refuses to run unless that directory is still owned by the user who ran sudo, mode 0700, and every file matches the
# checksums computed here on the PC.
ST=$(ssh $O $H 'mktemp -d /tmp/caldec-npu-stage.XXXXXXXX')
case $ST in /tmp/caldec-npu-stage.????????) ;; *) echo "unexpected staging dir: $ST" >&2; exit 1;; esac
TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
files=(runner/caldec_npu.py runner/service/caldec_npu_service.py runner/caldec_npu_server.c $AH
       runner/service/caldec-npu runner/service/caldec-npu.socket runner/service/caldec-npu.service
       $CK/tokenizer/tokenizer.json $CK/tokenizer/tokenizer_config.json $CK/rl_agent_config.json $RT/rknn_api.h $RT/librknnrt.so
       $P/${SET}_s384.rknn)
W="96 128 160 192 224 256 320"; for s in 512 640; do [ -f $P/${SET}_s${s}_now.rknn ] && W="$W $s"; done   # long-state buckets if built
for s in $W; do files+=($P/${SET}_s${s}_now.rknn); done
TABLES="tables_s384 $(for s in $W; do echo -n "tables_s$s "; done)"
{ for f in "${files[@]}"; do printf '%s  %s\n' "$(sha256sum < "$f" | cut -d' ' -f1)" "$(basename "$f")"; done
  for t in $TABLES; do (cd "$P" && sha256sum "$t"/*); done; } > "$TMPD/STAGED.sha256"
rsync -a -e "ssh $O" "${files[@]}" "$TMPD/STAGED.sha256" $H:$ST/
rsync -a -e "ssh $O" $(for t in $TABLES; do echo "$P/$t"; done) $H:$ST/
ssh $O $H "cd $ST && sha256sum *.rknn > SHA256SUMS && cat SHA256SUMS"
cat > $TMPD/caldec-npu-install.sh <<EOS
set -euo pipefail; ST=$ST; SET=\$1; D=/opt/caldec-npu; W="$W"
[ -d "\$ST" ] && [ ! -L "\$ST" ] || { echo "no staging dir \$ST" >&2; exit 1; }
[ "\$(stat -c %u "\$ST")" = "\${SUDO_UID:?run this with sudo}" ] && [ "\$(stat -c %a "\$ST")" = 700 ] \
  || { echo "\$ST is not private to the deploying user; not installing" >&2; exit 1; }
cd "\$ST"; sha256sum --quiet --strict -c STAGED.sha256 || { echo "staged files changed; not installing" >&2; exit 1; }
getent group caldec-clients >/dev/null || groupadd --system caldec-clients
id caldec >/dev/null 2>&1 || useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin --groups render caldec
install -d -m 0755 \$D
install -m 0644 caldec_npu.py caldec_npu_service.py caldec_npu_server.c act_head.npz tokenizer.json tokenizer_config.json rl_agent_config.json rknn_api.h SHA256SUMS \$D/
install -m 0755 librknnrt.so \$D/
install -m 0644 \${SET}_s*.rknn \$D/
for t in tables_s*; do install -d -m 0755 \$D/\$t; install -m 0644 \$t/* \$D/\$t/; done
{ echo \$((1 + \$(echo \$W | wc -w))); echo "384 \${SET}_s384.rknn tables_s384 0"; for s in \$W; do echo "\$s \${SET}_s\${s}_now.rknn tables_s\$s 1"; done; } > \$D/models.txt
gcc -O2 -o \$D/caldec_npu_server \$D/caldec_npu_server.c -I\$D -L\$D -lrknnrt -Wl,-rpath,'\$ORIGIN'
[ -x \$D/venv/bin/python ] || { python3 -m venv \$D/venv && \$D/venv/bin/pip install -q numpy==2.5.5 tokenizers==0.23.3; }
\$D/venv/bin/python -c "import numpy, tokenizers; print('numpy', numpy.__version__, 'tokenizers', tokenizers.__version__)"
install -m 0755 caldec-npu /usr/local/bin/caldec-npu
install -m 0644 caldec-npu.socket caldec-npu.service /etc/systemd/system/
systemctl daemon-reload; systemctl enable caldec-npu.socket; systemctl restart caldec-npu.socket
systemctl stop caldec-npu.service 2>/dev/null || true
echo "installed \$D (\$SET); socket: \$(systemctl is-active caldec-npu.socket)"
EOS
rsync -a -e "ssh $O" $TMPD/caldec-npu-install.sh $H:$ST/
echo "staged in $ST; now on the board: sudo bash $ST/caldec-npu-install.sh $SET"
