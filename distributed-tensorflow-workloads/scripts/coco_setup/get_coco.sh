#!/bin/bash

# This script gets the COCO dataset from the 'coco_urls.txt' file. Note that
# if you want to use a different version of the COCO datset, you should change
# the URLS in the 'coco_urls.txt' file

# Everything should be mounted under /mnt/coco-dataset. If you don't wish to
# moutn the file under /mnt/coco-dataset, then edit this variable
MOUNT_PATH="/mnt/tensorflow-files"

# Prepare to create folders for the COCO dataset
COCO_ROOT=${MOUNT_PATH}/coco-dataset
COCO_ZIP_FILES_PATH=${COCO_ROOT}/original-zip-files

# Get the COCO dataset from the URLs file. (Note that this 'coco_urls.txt'
# file is located under '/opt/coco-dataset/scripts/initial_setup' if using
# the Dockerfile.coco file provided. Otherwise, you will need to 'cd' to the
# proper directory before calling 'wget')
mkdir -p ${COCO_ZIP_FILES_PATH}
mv coco_urls.txt ${COCO_ZIP_FILES_PATH}
cd ${COCO_ZIP_FILES_PATH}
wget -i coco_urls.txt 

# Get a list of all the zip files that will ultimately be placed under the
# ${MOUNT_PATH} directory. The pretrained model is in a .tar.gz format, so
# we will have to later on 'grep' for the '.tar.gz' file (as there is only
# one of them).
coco_zip_files=$(ls *.zip)

# Unzip everything from current working directory to the new directory,
# ${COCO_ROOT}
for zip_file in ${coco_zip_files}; do
    unzip -o ${zip_file} -d ${COCO_ROOT}
done
