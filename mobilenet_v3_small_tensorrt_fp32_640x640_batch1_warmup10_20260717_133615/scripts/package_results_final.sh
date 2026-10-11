#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$RUN")"
NAME="$(basename "$RUN")"
ARCHIVE="$PARENT/${NAME}.publication_ready.tar.gz"
SIDECAR="$PARENT/${NAME}.publication_ready.archive_sha256.txt"
if [[ -e "$ARCHIVE" || -e "$SIDECAR" ]]; then
  echo "Refusing to overwrite publication archive or sidecar" >&2
  exit 3
fi
tar -czf "$ARCHIVE" -C "$PARENT" "$NAME"
sha256sum "$ARCHIVE" | tee "$SIDECAR"
echo "$ARCHIVE"
