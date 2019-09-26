#!/bin/sh
# Download pittsburg weights for tensorflow.

mkdir -p netvlad_tf_open/
wget http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip
unzip vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip
mv vd16_pitts30k_conv5_3_vlad_preL2_intra_white netvlad_tf_open/
rm vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip

echo 'model_checkpoint_path: "vd16_pitts30k_conv5_3_vlad_preL2_intra_white"' >> checkpoint
echo 'all_model_checkpoint_paths: "vd16_pitts30k_conv5_3_vlad_preL2_intra_white"' >> checkpoint
mv checkpoint netvlad_tf_open/vd16_pitts30k_conv5_3_vlad_preL2_intra_white
