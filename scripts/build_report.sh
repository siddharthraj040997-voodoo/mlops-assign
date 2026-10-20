#!/usr/bin/env bash
# Rebuild report/REPORT.pdf from report/REPORT.md (needs pandoc + wkhtmltopdf).
set -euo pipefail
cd "$(dirname "$0")/../report"
pandoc REPORT.md -f markdown -t html5 --standalone --metadata pagetitle="Heart Disease MLOps Report" \
  --css report.css --embed-resources -o REPORT.html
wkhtmltopdf --enable-local-file-access -q --page-size A4 \
  --margin-top 14mm --margin-bottom 14mm --margin-left 13mm --margin-right 13mm \
  --footer-center "[page] / [topage]" --footer-font-size 8 REPORT.html REPORT.pdf
echo "built report/REPORT.pdf"
