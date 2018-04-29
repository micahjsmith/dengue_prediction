#!/usr/bin/env bash
set -e
set -x

# activate environment :(
tox -e py36 --notest
source .tox/py36/bin/activate
which python
python --version

# actual testing
python -m dengue_prediction.data.sync_data download
python scripts/validate_pr.py
