#!/usr/bin/env bash
# Box-side half of the deploy: make /nvme/work/wmo-compressor a complete, self-contained
# service root and (re)start it. Idempotent; safe to re-run on every deploy.
#
# Never prints the bearer token. The token is generated here on first run and lives only in
# /etc/wmo-compressor/env (root-owned, 0600).
#
# Everything lives on /nvme because the box's root filesystem runs at ~97% full.
set -euo pipefail

ROOT=/nvme/work/wmo-compressor
ENV_DIR=/etc/wmo-compressor
ENV_FILE="${ENV_DIR}/env"
# The weights C1's GPU bench already pulled; copied in so this service does not depend on a
# bench scratch directory continuing to exist.
BENCH_HF=/nvme/work/c1-bench/hf/hub
PIP_CACHE=/nvme/work/c1-bench/pipcache
MODEL_DIR=models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank
PUBLIC_IP="${WMO_COMPRESSOR_PUBLIC_IP:-40.80.93.150}"

mkdir -p "${ROOT}/tls" "${ROOT}/hf/hub" "${ROOT}/tmp"
export TMPDIR="${ROOT}/tmp"

echo "[bootstrap] venv"
if [[ ! -x "${ROOT}/venv/bin/python" ]]; then
  python3 -m venv "${ROOT}/venv"
fi
"${ROOT}/venv/bin/pip" install --quiet --upgrade pip
# --cache-dir points at the wheels the GPU bench already downloaded, so the multi-GB torch
# install is a local unpack rather than a fresh download.
"${ROOT}/venv/bin/pip" install --quiet --cache-dir "${PIP_CACHE}" -r "${ROOT}/requirements.txt"

echo "[bootstrap] model weights"
if [[ ! -d "${ROOT}/hf/hub/${MODEL_DIR}" ]]; then
  cp -a "${BENCH_HF}/${MODEL_DIR}" "${ROOT}/hf/hub/${MODEL_DIR}"
fi

echo "[bootstrap] tls"
if [[ ! -f "${ROOT}/tls/cert.pem" ]]; then
  # Self-signed and pinned by clients: no domain to buy, no CA to trust, and a stolen
  # certificate is useless without the bearer token. SAN carries the box's public IP because
  # that is what clients dial.
  openssl req -x509 -newkey rsa:4096 -nodes -days 825 \
    -keyout "${ROOT}/tls/key.pem" -out "${ROOT}/tls/cert.pem" \
    -subj "/CN=wmo-compressor.h100-dev-box-6" \
    -addext "subjectAltName=IP:${PUBLIC_IP},DNS:wmo-compressor.h100-dev-box-6" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature,keyEncipherment" 2>/dev/null
  # CA:FALSE is deliberate and verified: clients pin this exact leaf as their entire trust
  # store (checked with both curl and Python httpx), so it never needs to sign anything.
  chmod 600 "${ROOT}/tls/key.pem"
  chmod 644 "${ROOT}/tls/cert.pem"
fi

echo "[bootstrap] token"
sudo mkdir -p "${ENV_DIR}"
sudo chmod 700 "${ENV_DIR}"
if ! sudo test -f "${ENV_FILE}"; then
  # Generated on the box, never transported, never echoed. Read it back only with
  # `sudo cat /etc/wmo-compressor/env` when a teammate needs to be handed access.
  printf 'WMO_COMPRESSOR_TOKEN=%s\n' "$(openssl rand -hex 32)" | sudo tee "${ENV_FILE}" >/dev/null
fi
sudo chmod 600 "${ENV_FILE}"
sudo chown root:root "${ENV_FILE}"

echo "[bootstrap] systemd"
sudo cp "${ROOT}/wmo-compressor.service" /etc/systemd/system/wmo-compressor.service
sudo systemctl daemon-reload
sudo systemctl enable wmo-compressor.service >/dev/null
sudo systemctl restart wmo-compressor.service

echo "[bootstrap] waiting for health"
for _ in $(seq 1 90); do
  if curl -sf --cacert "${ROOT}/tls/cert.pem" --resolve "wmo-compressor.h100-dev-box-6:8443:127.0.0.1" \
      "https://wmo-compressor.h100-dev-box-6:8443/healthz" >/dev/null 2>&1; then
    echo "[bootstrap] healthy"
    curl -s --cacert "${ROOT}/tls/cert.pem" --resolve "wmo-compressor.h100-dev-box-6:8443:127.0.0.1" \
      "https://wmo-compressor.h100-dev-box-6:8443/healthz"
    echo
    exit 0
  fi
  sleep 2
done

echo "[bootstrap] FAILED to become healthy; last 40 journal lines:" >&2
sudo journalctl -u wmo-compressor -n 40 --no-pager >&2
exit 1
