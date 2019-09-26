
Code from: 
- https://github.com/google/compare\_gan.git
- https://github.com/Nanne/pytorch-NetVlad

# Known issues

## 1

- Pb :

        ValueError: Object arrays cannot be loaded when allow_pickle=False

- Sol: it is an issue from numpy==1.16.4 Downgrade to 1.16.2

        sudo pip3 install numpy==1.16.2


If you use this paper, cite the wasabi paper.
