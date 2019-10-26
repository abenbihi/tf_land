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

Get the VGG-NetVLAD weights trained on Pittsburg [3]
```bash
cd netvlad/meta/weights/
./get_weights.sh
```

# Training on Extended-CMU-Seasons
The following instructions show how to train NetVLAD on specific slices of the
Extended-CMU-Sesons dataset. The intrusctions can be generalized to any other
slice and other datasets.

Get the dataset (this can take some time, so go grab some coffee).
```bash
cd netvlad/datasets/
./Extended-CMU-Seasons.sh


Split the slice
TODO

# Known issues

## 1

- Pb :

        ValueError: Object arrays cannot be loaded when allow_pickle=False

- Sol: it is an issue from numpy==1.16.4 Downgrade to 1.16.2

        sudo pip3 install numpy==1.16.2



