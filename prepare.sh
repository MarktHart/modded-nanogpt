#!/usr/bin/env bash

set -e

sudo docker build . & echo "Start downloading docker in background"
python3 -m venv env
source env/bin/activate
pip install -r data/requirements.txt
python data/cached_fineweb10B.py 40
