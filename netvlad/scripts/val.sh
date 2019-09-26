#!/bin/sh


MACHINE=1
if [ "$MACHINE" -eq 0 ]; then
  ws_dir=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  ws_dir=/home/gpu_user/assia/ws/
else
  echo "Error in train.sh: Get your MTF MACHINE macro correct"
  exit 1
fi

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

if ! [ -d res/"$trial"/log/val/ ]; then
  mkdir -p res/"$trial"/log/val/
fi


python3 val.py \
  --trial "$trial" \
  --data_id "$data_id" \
  --mean_fn meta/mean_std.txt \
  --split_dir "$ws_dir"/datasets/CMU-Seasons\
  --img_dir "$ws_dir"/datasets/Extended-CMU-Seasons \
  --batch_size 3 \
  --resize 1 \
  --h 384 \
  --w 512 \
  --no_finetuning 0 \
  --moving_average_decay 0.9999 
