#!/usr/bin/env bash
# Laptop-side deploy: push this directory to h100-dev-box-6 and restart the service.
#
#     ./deploy.sh              # rsync + bootstrap + restart + health check
#     ./deploy.sh --cert-only  # just refetch the public cert (after a cert rotation)
#
# Requires plain `ssh h100-dev-box-6` to work (the az ssh AAD path is broken on this fleet).
# Prints no secrets: the bearer token is generated on the box and never leaves it.
set -euo pipefail

HOST="${WMO_COMPRESSOR_HOST:-h100-dev-box-6}"
ROOT=/nvme/work/wmo-compressor
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fetch_cert() {
  # The pinned public certificate is committed so clients can verify without a CA.
  scp -q "${HOST}:${ROOT}/tls/cert.pem" "${HERE}/compressor-cert.pem"
  echo "[deploy] pinned cert -> ${HERE}/compressor-cert.pem"
}

if [[ "${1:-}" == "--cert-only" ]]; then
  fetch_cert
  exit 0
fi

echo "[deploy] rsync -> ${HOST}:${ROOT}"
ssh "${HOST}" "mkdir -p ${ROOT}"
rsync -az --delete-excluded \
  --include='server.py' \
  --include='requirements.txt' \
  --include='wmo-compressor.service' \
  --include='bootstrap.sh' \
  --exclude='*' \
  "${HERE}/" "${HOST}:${ROOT}/"

echo "[deploy] bootstrap on ${HOST}"
ssh "${HOST}" "chmod +x ${ROOT}/bootstrap.sh && ${ROOT}/bootstrap.sh"

fetch_cert
echo "[deploy] done. Verify from here:"
echo "  curl --cacert ${HERE}/compressor-cert.pem --resolve wmo-compressor.h100-dev-box-6:8443:40.80.93.150 \\"
echo "    https://wmo-compressor.h100-dev-box-6:8443/healthz"
