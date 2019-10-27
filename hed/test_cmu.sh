#!/bin/sh

if [ $# -eq 0 ]; then
    echo "Usage"
    echo "1. slice_id"
    echo "2. cam_id"
    echo "3. survey_id"
    exit 0
fi

if [ $# -ne 3 ]; then
    echo "Bad number of arguments"
    echo "1. slice_id"
    echo "2. cam_id"
    echo "3. survey_id"
    exit 1
fi

slice_id="$1"
cam_id="$2"
survey_id="$3"

IMG_DIR=datasets/Extended-CMU-Seasons-Undistorted/
res_dir=res/cmu/slice"$slice_id"
if [ -d "$res_dir" ]; then
    mkdir -p "$res_dir"
fi

python3 test_cmu.py \
    --img_dir "$IMG_DIR" \
    --res_dir "$res_dir" \
    --slice_id "$slice_id" \
    --cam_id "$cam_id" \
    --survey_id "$survey_id" \
    --mean_file ./list/mean_vgg.txt
