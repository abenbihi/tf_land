#!/bin/sh

if [ "$#" -eq 0 ]; then
  echo "1. trial"
  echo "2. data_id"
  exit 0
fi

if [ "$#" -ne 2 ]; then
  echo "Error: bad number of arguments"
  echo "1. trial"
  echo "2. data_id"
  exit 1
fi

trial="$1"
data_id="$2"

if ! [ -d res/"$trial"/cache/ ]; then
  mkdir -p res/"$trial"/cache/
fi

split_dir=meta/data_splits/cmu/
img_dir=meta/data/cmu/

python3 -m tools.data_loader \
  --trial "$trial" \
  --data_id "$data_id" \
  --split_dir "$split_dir" \
  --img_dir "$img_dir" \
  --mean_fn meta/mean_std.txt \
  --batch_size 3 \
  --resize 1 \
  --h 384 \
  --w 512 \
  --margin 0.1 \
  --N_nh 2 \
  --N_nr 10
 
