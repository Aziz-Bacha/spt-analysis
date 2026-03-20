# Critical Analysis of Superpoint Transformer (SPT)

NPM3D Project - MVA 2026 - Aziz Bacha

This repository contains the experiment code for the critical analysis of
*Efficient 3D Semantic Segmentation with Superpoint Transformer* (Robert et al., ICCV 2023).

## Setup

1. Clone and install the SPT repository:
```bash
git clone https://github.com/drprojects/superpoint_transformer.git
cd superpoint_transformer
./install.sh
```

2. Download the pretrained DALES checkpoint from [Zenodo](https://zenodo.org/records/8042712) and place it in `models/`.

3. Download the datasets:
   - **DALES**: Register at the [DALES request form](https://docs.google.com/forms/d/e/1FAIpQLSdl2Tag498Goc3yFPLllya3ICorYlcBcw3IQi9Lw-Wj9ow2jQ/viewform) and download `DALESObjects.tar.gz`
   - **Vancouver**: download a tile from [Vancouver LiDAR 2022](https://opendata.vancouver.ca/explore/dataset/lidar-2022/)
   - **Toronto-3D**: download from [GitHub](https://github.com/WeikaiTan/Toronto-3D)

4. Copy the notebooks and scripts into the SPT project:
```bash
cp notebooks/*.ipynb /path/to/superpoint_transformer/notebooks/
cp scripts/run_partition.py /path/to/superpoint_transformer/
```

## Experiments

| Notebook | Description |
|----------|-------------|
| `dales_evaluation.ipynb` | Reproduction on DALES: per-class IoU, confusion matrix, purity analysis, superpoint size vs accuracy |
| `vancouver_inference.ipynb` | Cross-domain transfer: DALES model on Vancouver, building confusion analysis |
| `toronto3d_inference.ipynb` | Cross-domain transfer: DALES model on Toronto-3D |
| `partition_sensitivity.ipynb` | Regularization sweep and feature ablation on Vancouver |

The `scripts/run_partition.py` script is used by the partition sensitivity notebook to run each configuration in a separate subprocess (to avoid GPU memory accumulation).

## Acknowledgments

The notebooks in this repository build upon the tutorial notebooks
provided in the [SPT repository](https://github.com/drprojects/superpoint_transformer/tree/master/notebooks),
extending them with custom evaluation, cross-domain transfer, and ablation experiments.

## References

- **Paper**: [Efficient 3D Semantic Segmentation with Superpoint Transformer](https://arxiv.org/abs/2306.08045) (ICCV 2023)
- **Original code**: [github.com/drprojects/superpoint_transformer](https://github.com/drprojects/superpoint_transformer)
