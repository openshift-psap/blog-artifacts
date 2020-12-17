#!/bin/bash

# This script runs the TensorFlow 'models' image detection task

# First input is the number of GPUs
NUM_GPUS=$1

# Second input is the mount path. Default would be:
# MOUNT_PATH=/mnt/tensorflow-files
MOUNT_PATH=$2

# Set TensorFlow related directories
IMAGE_DETECTION_ROOT=${MOUNT_PATH}
TENSORFLOW_MODELS=${IMAGE_DETECTION_ROOT}/models
TENSORFLOW_RECORD_DIR=${IMAGE_DETECTION_ROOT}/tf-records

# Set 'train_file_pattern', 'eval_file_pattern', and finally the
# 'val_json_file'
TRAIN_FILE_PATTERN="${TENSORFLOW_RECORD_DIR}/train-*"
EVAL_FILE_PATTERN="${TENSORFLOW_RECORD_DIR}/val-*"
VAL_JSON_FILE="${TENSORFLOW_RECORD_DIR}/instances_val2017.json"

# Create directories that don't already exist
mkdir -p ${TENSORFLOW_RECORD_DIR}

# Set retinanet yaml
RETINANET_YAML=${TENSORFLOW_RECORD_DIR}/retina-net.yaml

# Create retinanet yaml:
cat <<EOF > ${RETINANET_YAML}
type: 'retinanet'
train:
  train_file_pattern: ${TRAIN_FILE_PATTERN}
eval:
  eval_file_pattern: ${EVAL_FILE_PATTERN}
  val_json_file: ${TENSORFLOW_RECORD_DIR}/instances_val2017.json
EOF

## Set the parameters
python3 ${TENSORFLOW_MODELS}/official/vision/detection/main.py \
  --strategy_type=multi_worker_mirrored \
  --num_gpus=${NUM_GPUS} \
  --model_dir="${TENSORFLOW_MODELS}" \
  --mode=train \
  --config_file="${RETINANET_YAML}"
