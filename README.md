# Graph Mixup on Approximate Gromov–Wasserstein Geodesics

## Overview
Implementation of [Graph Mixup on Approximate Gromov–Wasserstein Geodesics](https://proceedings.mlr.press/v235/zeng24e.html) in ICML 2024
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

## Reference
If you find this paper helpful to your research, please kindly cite the following paper:
```

@InProceedings{pmlr-v235-zeng24e,
  title = 	 {Graph Mixup on Approximate Gromov–{W}asserstein Geodesics},
  author =       {Zeng, Zhichen and Qiu, Ruizhong and Xu, Zhe and Liu, Zhining and Yan, Yuchen and Wei, Tianxin and Ying, Lei and He, Jingrui and Tong, Hanghang},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {58387--58406},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/zeng24e/zeng24e.pdf},
  url = 	 {https://proceedings.mlr.press/v235/zeng24e.html},
  abstract = 	 {Mixup, which generates synthetic training samples on the data manifold, has been shown to be highly effective in augmenting Euclidean data. However, finding a proper data manifold for graph data is non-trivial, as graphs are non-Euclidean data in disparate spaces. Though efforts have been made, most of the existing graph mixup methods neglect the intrinsic geodesic guarantee, thereby generating inconsistent sample-label pairs. To address this issue, we propose GeoMix to mixup graphs on the Gromov-Wasserstein (GW) geodesics. A joint space over input graphs is first defined based on the GW distance, and graphs are then transformed into the GW space through equivalence-preserving transformations. We further show that the linear interpolation of the transformed graph pairs defines a geodesic connecting the original pairs on the GW manifold, hence ensuring the consistency between generated samples and labels. An accelerated mixup algorithm on the approximate low-dimensional GW manifold is further proposed. Extensive experiments show that the proposed GeoMix promotes the generalization and robustness of GNN models.}
}

```
