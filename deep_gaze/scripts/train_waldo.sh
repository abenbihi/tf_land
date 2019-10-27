#!/bin/sh

LOG_DIR=./log
MACHINE=2
dataset_id=0

if [ "$MACHINE" -eq 0 ]; then # baby
  WS_DIR=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then # gpu
  WS_DIR=/home/gpu_user/assia/ws/
elif [ "$MACHINE" -eq 2 ]; then
  WS_DIR=/opt/BenbihiAssia/
fi

DATASET_DIR="$WS_DIR"datasets/waldo/hider_sal/


if [ $# -eq 0 ]; then
  echo "Usage"
  echo "  1. xp_name "
  echo "  2. dataset_id "
  exit 0
fi

if [ $# -ne 2 ]; then
  echo "Bad number of arguments"
  exit 1
fi


xp_name="$1"
dataset_id="$2"


log_dir=""$LOG_DIR"/"$xp_name"/"
if [ -d "$log_dir" ]; then
  while true; do
    read -p ""$log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
    case $yn in
      [Yy]* ) rm -rf "$log_dir"; mkdir -p "$log_dir"; break;;
      [Nn]* ) exit;;
      * ) * echo "Please answer yes or no.";;
    esac
  done
else
  mkdir -p "$log_dir"
fi


if [ 1 -eq 1 ]; then
  python3 -m waldo.train_512 \
    --dataset_dir "$DATASET_DIR" \
    --dataset_id "$dataset_id" \
    --w 512 \
    --h 512 \
    --resize_img 0 \
    --mean_file meta/mean_vgg.txt \
    --xp_name "$xp_name" \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-1 \
    --adam_b1 0.9 \
    --adam_b2 0.999 \
    --adam_eps 1e-08 \
    --moving_average_decay 0.9999 \
    --activ_fn leaky_relu \
    --leaky_alpha 0.2 \
    --train_log_dir "$LOG_DIR" \
    --display_interval 100 \
    --summary_interval 500 \
    --save_interval 10000000 \
    --start 1
fi


if [ 0 -eq 1 ]; then
  python3 -m waldo.train \
    --dataset_dir "$DATASET_DIR" \
    --dataset_id "$dataset_id" \
    --resize_img 1 \
    --w 1024 \
    --h 1024 \
    --xp_name "$xp_name" \
    --epochs 1 \
    --batch_size 1 \
    --lr 1e-10 \
    --adam_b1 0.9 \
    --adam_b2 0.999 \
    --adam_eps 1e-08 \
    --moving_average_decay 0.9999 \
    --train_log_dir "$LOG_DIR" \
    --display_interval 1 \
    --summary_interval 10 \
    --save_interval 10000000 \
    --start 1
fi


