#!/bin/bash
cd "$(dirname "$0")"
pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q --break-system-packages
python3 app.py
