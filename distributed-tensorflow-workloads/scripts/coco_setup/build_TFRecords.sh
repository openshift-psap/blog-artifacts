#!/bin/bash

# This script generates TFRecord files for the COCO dataset

# Everything should be mounted under /mnt/coco-dataset. If you don't wish to mount
# the file under /mnt/tensorflow-files, then edit this variable
MOUNT_PATH="/mnt/tensorflow-files"

# Where our unzipped COCO dataset is
COCO_ROOT=${MOUNT_PATH}/coco-dataset

# COCO dataset directory hierarchy
COCO_ANNOTATIONS_DIR=${COCO_ROOT}/annotations
COCO_TFRECORD_DIR=${MOUNT_PATH}/tf-records

# Get TensorFlow models and create the TFRecord file
TENSORFLOW_MODELS=${MOUNT_PATH}/models
git clone https://github.com/tensorflow/models.git ${TENSORFLOW_MODELS}
cd ${TENSORFLOW_MODELS}/research/object_detection/dataset_tools
python3 create_coco_tf_record.py \
	--logtostderr \
        --train_image_dir=${COCO_ROOT}/train2017 \
        --val_image_dir=${COCO_ROOT}/val2017 \
        --test_image_dir=${COCO_ROOT}/test2017 \
        --train_annotations_file=${COCO_ANNOTATIONS_DIR}/instances_train2017.json \
        --val_annotations_file=${COCO_ANNOTATIONS_DIR}/instances_val2017.json \
        --testdev_annotations_file=${COCO_ANNOTATIONS_DIR}/image_info_test-dev2017.json \
        --output_dir=${COCO_TFRECORD_DIR}
