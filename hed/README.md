Tensorflow port of the caffe model provided by the authors of 
```bash
@inproceedings{xie2015holistically,
    title={Holistically-nested edge detection},
    author={Xie, Saining and Tu, Zhuowen},
    booktitle={Proceedings of the IEEE international conference on computer
        vision},
    pages={1395--1403},
    year={2015}
}
```

# Installation
Get the code
```bash
git clone https://gitlab.georgiatech-metz.fr/paper/tf_land.git
```

Most of the python3 packages are 'standard' i.e. you probably already have them
installed on your computer already. Anyway, I provide you with the list of used
packages in `requirements.txt`. Either find the one you are missing manually in
`requirements.txt` and install them; or install them all with 
```bash
pip3 install -r requirements.txt
```

Get the hed weights trained on Cityscapes.
```bash
cd meta/weights/
./get_weights.sh
```

# Training
## Prepare the data
Generate two files `train.txt` and `val.txt` with the list of images relative
path name belonging to the train/val set.

## Data
Gnera
- Download the dataset you want to train on
- Decide on the split 
- Generate the list of images for `train.txt` and `val.txt`

For example:
    $ head -n4 train.txt
    04/image_2/000225.png
    04/image_2/000226.png
    04/image_2/000227.png
    04/image_2/000228.png


for a dataset structure like this

    $HOME
    |
    |_data
      |_img
        |_  0
            |_0000.png
            |_0001.png
      |_edge
        |_  0
            |_0000.png
            |_0001.png

## Dummy ground truth edges
You can generate dummy ground truth edges with canny with the script in
`dummy_data/gen_data.py`. Set the path to your images in the script (`IMG_ROOT_DIR`).

    cd dummy_data
    python dummy_data

It stores this dummy data in `dummy_data/canny`.

## Train

In `train.sh`, set the data path

    IMG_DIR='/path/to/data/img/'
    EDGE_DIR='/path/to/data/edge/'

Then run it

    ./train.sh xp0 0 4 2

  
As it runs, you can check the tensorboad logs (train and val) with 

    tensorboard --logdir log/xp0/
