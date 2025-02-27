#!/bin/sh

DATASET_PATH=../DATASET_AMOS

export PYTHONPATH=./
export RESULTS_FOLDER=output_amos
export hymavi_preprocessed="$DATASET_PATH"/hymavi_raw/hymavi_raw_data/Task08_AMOS
export hymavi_raw_data_base="$DATASET_PATH"/hymavi_raw

python hymavi/run/run_training.py 3d_fullres hymavi_trainer_amos 8 0
