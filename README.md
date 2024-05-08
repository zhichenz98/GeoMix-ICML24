# Graph Mixup on Approximate Gromov–Wasserstein Geodesics

## Overview
Implementation of Graph Mixup on Approximate Gromov–Wasserstein Geodesics in ICML 2024
<p align="center">
  <img width="800" height="500" src="./imgs/geomix.png">
</p>

## Environment Setup
```
    conda create -n geomix python=3.12
    conda activate geomix
    conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install torch_geometric
    conda install -c conda-forge pot
    conda install matplotlib
```

## Run the code
```
    python src/train.py --data MUTAG --model GCN --num_node 20 --augment True
```

- ```--data```: select from ```IMDB-BINARY | IMDB-MULTI | MUTAG | PROTEINS | MSRC_9```.
- ```--model```: select from ```GCN | GIN | APPNP```.
- ```--num_node```: size of the mixup graph (```20``` for IMDB/MUTAG, ```40``` for PROTEINS/MSRC_9).
- ```--augment```: ```True``` to perform GeoMix, ```False``` to use backbone vanilla models.





## Dataset

| Dataset  | # Graphs | # Nodes | # Edges | # Features | # Class |
|----------|----------|---------|---------|------------|---------|
| PROTEINS |    1,113 |   43.31 |   77.79 |          1 |       2 |
| MUTAG    |      188 |   17.93 |   19.79 | None       |       2 |
| MSRC-9   |      221 |   40.58 |   97.94 | None       |       8 |
| IMDB-B   |    1,000 |   19.77 |   96.53 | None       |       2 |
| IMDB-M   |    1,500 |   12.74 |   53.88 | None       |       3 |


## Mixup graph visualization
<p align="center">
  <img width="900" height="480" src="./imgs/example.png">
</p>
