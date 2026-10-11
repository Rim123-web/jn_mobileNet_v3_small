#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="${RUN}.tar.gz"
test "$(python3 -c "import json; print(json.load(open('$RUN/validation/final_validation_report.json'))['status'])")" = PASS
tar -czf "$ARCHIVE" -C "$(dirname "$RUN")" "$(basename "$RUN")"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
