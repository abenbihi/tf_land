#!/bin/sh

LOG_DIR=log/
IMG_DIR='/mnt/dataX/assia/kitti/dataset/sequences/'
EDGE_DIR='dummy_data/canny/'

if [ "$1" = '-h' ]; then 
  echo "Usage"
  echo "  1. Xp name"
  echo "  2. Number of epochs to train"
  echo "  3. Number of epochs between tests"
  exit 0
fi

if [ "$#" -ne 3 ]; then
  echo "Error: bad number of arguments"
  echo "Usage"
  echo "  1. Xp name"
  echo "  2. Number of epochs to train"
  echo "  3. Number of epochs between tests"
  exit 1
fi

xp_name="$1"
max_train_epoch="$2" # 100
eval_interval_epoch="$3" # 20 
num_iter=$((max_train_epoch/eval_interval_epoch)) # 5

log_dir=""$LOG_DIR""$xp_name""
echo "IMG_DIR: "$IMG_DIR"/"
echo "edge_dir: "$EDGE_DIR""
echo "log_dir: "$log_dir""
echo "Train for "$max_train_epoch" epochs"
echo "Test every "$eval_interval_epoch" epochs"


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

# Run the bench
if [ 1 -eq 1 ]; then
  i=0
  echo "num_iter = "$num_iter""
  while [ "$i" -lt "$num_iter" ] # 5
  do

    python train.py \
      --img_dir "$IMG_DIR" \
      --edge_dir "$EDGE_DIR" \
      --mean_file ./meta/mean_vgg.txt \
      --patch_h 256 \
      --patch_w 256 \
      --xp_name "$xp_name" \
      --log_dir "$LOG_DIR" \
      --display_interval 10 \
      --summary_interval 10 \
      --save_interval 10000000 \
      --epochs "$eval_interval_epoch" \
      --batch_size 4 \
      --lr 1e-5 \
      --adam_b1 0.9 \
      --adam_b2 0.999 \
      --adam_eps 1e-08 \
      --moving_average_decay 0.9999 \
      --start 1

    if [ $? -ne 0 ]; then
      echo "Error in training "$i" "
      exit 1
    fi

    python ""$CODE_ROOT_DIR"eval.py" \
      --img_dir "$IMG_DIR" \
      --edge_dir "$EDGE_DIR" \
      --mean_file ./meta/mean_vgg.txt \
      --patch_h 256 \
      --patch_w 256 \
      --batch_size 4 \
      --xp_name "$xp_name" \
      --log_dir "$LOG_DIR"

    if [ $? -ne 0 ]; then
      echo "Error in eval "$i" "
      exit 1
    fi

    i=$((i+1))
  done
fi

if [ 0 -eq 1 ]; then
  # test the hed caffe
  python ""$CODE_ROOT_DIR"test.py" \
    --img_dir "$IMG_DIR" \
    --edge_dir "$edge_dir" \
    --mean_file ./meta/mean_vgg.txt \
    --patch_h 384 \
    --patch_w 384 \
    --xp_name "$xp_name" \
    --log_dir "$LOG_DIR"
fi
