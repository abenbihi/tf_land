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

log_dir=res/"$trial"/
if [ -d "$log_dir" ]; then
    while true; do
        read -p ""$log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
        case $yn in
            [Yy]* ) 
              rm -rf "$log_dir"; 
              mkdir -p "$log_dir"/log/train/;
              mkdir -p "$log_dir"/cache/;
              break;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
  mkdir -p "$log_dir"/log/train/;
  mkdir -p "$log_dir"/cache/;
  mkdir -p "$log_dir";
fi

split_dir=meta/data_splits/cmu/
img_dir=meta/data/cmu/

python3 train.py \
  --trial "$trial" \
  --data_id "$data_id" \
  --mean_fn meta/mean_std.txt \
  --split_dir "$split_dir" \
  --img_dir "$img_dir" \
  --batch_size 3 \
  --resize 1 \
  --h 384 \
  --w 512 \
  --N_nr 10 \
  --N_nh 3 \
  --num_clusters 64 \
  --margin 1 \
  --n_epochs 30 \
  --C 1000 \
  --lr 0.001 \
  --moving_average_decay 0.9999 \
  --optim SGD \
  --momentum 0.9 \
  --weightDecay 0.001 \
  --num_epochs_per_decay 5 \
  --lr_decay_factor 0.5 \
  --seed 123 \
  --log_interval 50 \
  --summary_interval 200 \
  --save_interval 2 \
  --start_train 1
  
