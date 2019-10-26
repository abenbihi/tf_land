# Citation
If you use this code, please cite the following paper:
```
@article{benbihi2019image,
  title={Image-Based Place Recognition on Bucolic Environment Across Seasons
    From Semantic Edge Description},
  author={Benbihi, Assia and Geist, Matthieu and Pradalier, C{\'e}dric},
  journal={Preprint},
  year={2019}
}
```

# Credits
This repository integrates code from 
- [https://github.com/google/compare\_gan.git](https://github.com/google/compare\_gan.git) [1]
- [https://github.com/Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad) [2]
- [https://github.com/uzh-rpg/netvlad_tf_open](https://github.com/uzh-rpg/netvlad_tf_open) [3]


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

Get the VGG-NetVLAD weights trained on Pittsburg [3]
```bash
cd netvlad/meta/weights/
./get_weights.sh
```

# Training on Extended-CMU-Seasons
The following instructions show how to train NetVLAD on specific slices of the
Extended-CMU-Sesons dataset. The intrusctions can be generalized to any other
slice and other datasets.

Get the dataset (this can take some time, so go grab some coffee) and the meta
data.
```bash
cd meta/data/cmu/
./Extended-CMU-Seasons.sh

cd meta/data/cmu
./get_survey_list.sh
```

## Generate the data splits
### Long explanation of the data splits.
- To run image retrieval on images, you
need two sets of images. The first one is the `database` set from which you
want to retrieve an image. The second one is the `query` set for which you want
to find the most similar image in the `database`. 
- When you train a system, you need train/validation/test image sets. 
- When you train an image retrieval system, you still have to generate
  train/val/test splits for your evaluation to be statiscally sound. But each
  of these three splits must be divided into `database`/`query` sets. 
- Obvisouly, the `database`/`query` set hold images of the same location (else
  you would never retrieve similar images). BUT, the train/val/splits must hold
  images of disjoint locations to maintain the soudness of your evaluation.
  Else, you could train a system to work well on a location even though the
  location image from the train set and the validation set are taken under
  different conditions.
- Remark: this means that you need to the camera pose of the images to build
  these splits.

### Short explanation of the data splits.
You need three disjoint data splits to form the train/val/test.
Each of these splits is divided into `database`/`query` sets.
You need the camera pose to generate these splits. (For more info, check the
paragraph above)

The `tools/cmu_splitter.py` script provides an example on how to split the
data on the park section on the Extended-CMU-Seasons dataset for which we have
ground-truth poses (i.e. slices [22,25]). Each section is traversed 11 times
and I call survey the images collected from one traversal. (There actually are
12 traversals but the last one is always too fast/short to be used here.)

The current `tools/cmu_splitter.py` that traversals 5 to 9 are used as
query sets for all splits and the rest are used in the database. The slices
22,23 are used for train, 24 for validation and 25 for test.

### Run
Generate the lists of images for each split
```bash
python -m tools.cmu_splitter
```
This saves the data lists in `meta/data_splits/cmu/0/`. To generate other data
lists, change the `data_id` variable in the `tools/cmu_splitter.py` and run the
script again. The results will be stored in `meta/data_splits/cmu/<data_id>/`

## Train
```bash
./scripts/train.sh 0 0
./scripts/train.sh <trial> <data id>
```
The training logs and the models are stored in `res/0/log/train/`. You can
visualise them with tensorboard with 
```bash
tensorboard --logdir res/0/log/train/
```

## Evaluate
Compute the evaluation metrics recall@N and mean Average Precision (mAP) on the
trained network.
```bash
./scripts/val.sh 0 0
```

To evaluate the NetVLAD trained on the Pittsburd dataset, set the
`no_finetuning` option to in `./scripts/val.sh`.


# Known issues
## 1
Error message
```bash
ValueError: Object arrays cannot be loaded when allow_pickle=False
```
Solution: it is an issue from numpy==1.16.4 Downgrade to 1.16.2
```bash
sudo pip3 install numpy==1.16.2
```



