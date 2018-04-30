#!/usr/bin/env bash
set -e
set -x

# setup environment
tox --develop --notest
tox_py_env=$(python -c 'from tox_travis.envlist import guess_python_env;print(guess_python_env())')
source ".tox/${tox_py_env}/bin/activate"
which python
python --version

# actual testing
python -m dengue_prediction.data.sync_data download
python scripts/validate_pr.py
