#!/bin/sh

LOG_DIR=log/
IMG_DIR='/mnt/data_drive/dataset/lake/Dataset/'


#if [ -d "$log_dir" ]; then
#  while true; do
#    read -p ""$log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
#    case $yn in
#      [Yy]* ) rm -rf "$log_dir"; mkdir -p "$log_dir"; break;;
#      [Nn]* ) exit;;
#      * ) * echo "Please answer yes or no.";;
#    esac
#  done
#else
#  mkdir -p "$log_dir"
#fi

# Run the bench
  
python3 ""$CODE_ROOT_DIR"test_lake.py" \
  --img_dir "$IMG_DIR" \
  --mean_file ./list/mean_vgg.txt

if [ $? -ne 0 ]; then
  echo "Error in eval "$i" "
  exit 1
fi



