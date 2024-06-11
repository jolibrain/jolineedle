#!/bin/bash

WORK_DIR="test_dir/"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "${WORK_DIR}"
python3 -m pytest -p no:cacheprovider -s tests --work_dir "${WORK_DIR}" "$*"
