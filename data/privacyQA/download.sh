#!/usr/bin/env bash

URL_PREFIX=https://media.githubusercontent.com/media/AbhilashaRavichander/PrivacyQA_EMNLP/master/data
curl -OL ${URL_PREFIX}/policy_train_data.csv
curl -OL ${URL_PREFIX}/policy_test_data.csv
curl -OL ${URL_PREFIX}/meta-annotations/OPP-115%20Annotations/train_opp_annotations.csv
curl -OL ${URL_PREFIX}/meta-annotations/OPP-115%20Annotations/test_opp_annotations.csv

python -W ignore process.py
