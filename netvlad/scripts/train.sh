#!/bin/sh


MACHINE=2
if [ "$MACHINE" -eq 0 ]; then
  ws_dir=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  ws_dir=/home/gpu_user/assia/ws/
elif [ "$MACHINE" -eq 2 ]; then
  ws_dir=/opt/BenbihiAssia/ws/
else
  echo "Error in train.sh: Get your MTF MACHINE macro correct"
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "1. trial"
  echo "2. data_id"
  echo "3. lr"
  exit 0
fi

if [ "$#" -ne 3 ]; then
  echo "Error: bad number of arguments"
  echo "1. trial"
  echo "2. data_id"
  echo "3. lr"
  exit 1
fi

trial="$1"
data_id="$2"
lr="$3"

#if ! [ -d res/"$trial"/cache/ ]; then
#  mkdir -p res/"$trial"/cache/
#fi

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

#split_dir="$ws_dir"/life_saver/datasets/CMU-Seasons/meta/retrieval/
#img_dir="$ws_dir"/datasets/Extended-CMU-Seasons/

split_dir= "$ws_dir"/datasets/CMU-Seasons
img_dir="$ws_dir"/datasets/Extended-CMU-Seasons
#lr=0.0001

python3 train.py \
  --trial "$trial" \
  --data_id "$data_id" \
  --model alexnet \
  --mean_fn meta/mean_std.txt \
  --batch_size 3 \
  --N_nr 10 \
  --N_nh 3 \
  --num_clusters 64 \
  --margin 1 \
  --n_epochs 30 \
  --C 1000 \
  --lr "$lr" \
  --moving_average_decay 0.9999 \
  --adam_b1 0.9 \
  --adam_b2 0.999 \
  --adam_eps 1e-8 \
  --optim SGD \
  --momentum 0.9 \
  --weightDecay 0.001 \
  --num_epochs_per_decay 5 \
  --lr_decay_factor 0.5 \
  --resize 1 \
  --h 384 \
  --w 512 \
  --seed 123 \
  --log_interval 50 \
  --summary_interval 200 \
  --save_interval 2 \
  --start_train 1
  
