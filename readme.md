# DiagNO: Diagonal Spherical Neural Operators for Heterogeneous Earth Dynamics Modeling

**Official implementation of DiagNO — NeurIPS 2026 Main Track (Oral)**

This repository provides the official implementation of **DiagNO (Diagonal Spherical Neural Operator)**, a neural operator designed for heterogeneous Earth dynamics modeling. It includes the model architecture, training and inference pipelines, experimental configurations, and utilities for spherical fluid dynamics simulations.

DiagNO is evaluated on simulated Spherical Shallow Water Equations (SSWE) and real-world Earth system reanalysis datasets, including ERA5 atmospheric data and GLORYS12 oceanic data.

## Overview

Conventional rotation-equivariant spherical neural operators typically restrict spectral modes to isolated evolution, limiting their ability to capture heterogeneous Earth dynamics. **DiagNO** addresses this limitation by explicitly modeling cross-modal interactions in the spherical spectral domain.

By exploiting the unique diagonal structure of the spherical harmonic spectrum, DiagNO introduces **intra- and inter-diagonal interaction mechanisms** to efficiently characterize zonal and meridional dynamics. This structured spectral modeling approach enables accurate and physically consistent forecasting across diverse geophysical scenarios.

The framework is evaluated on three representative tasks:

- **Spherical Shallow Water Equations (SSWE):** Simulated global fluid dynamics forecasting.
- **Ocean Surface Forecasting (GLORYS12):** Prediction of sea surface height, sea surface temperature, and ocean currents.
- **Atmospheric Forecasting (ERA5):** Global atmospheric prediction using geopotential and wind fields.

## Quick Start

To initiate the training pipeline, ensure that the required environment has been configured (see [Environment Setup](#environment-setup)).

The provided training script manages model configurations and launches the distributed training pipeline using PyTorch Distributed Data Parallel (DDP).

```bash
bash run_train.sh
```

Before running the script, please update the dataset paths, output directories, and GPU configurations according to your local environment.

## Configuration Guide

The project separates training configurations from model architecture definitions.

### 1. Training Parameters (`run_train.sh`)

Global execution settings and training hyperparameters are configured in the training shell script.

- **`ROOT_PATH`**: Output directory for model checkpoints, training logs, and statistics.
- **`MODELS`**: A space-separated list of model variants to train sequentially (e.g., `diagno_e128`).
- **`EPOCHS`**: Training duration, including `PRETRAIN_EPOCHS` for one-step pretraining and `FINETUNE_EPOCHS` for two-step autoregressive fine-tuning.
- **Learning Rates**: Independent learning rates for pretraining and fine-tuning.
- **Hardware Setup**: Configure `CUDA_VISIBLE_DEVICES` and `nproc_per_node` according to the available GPU resources.

Please ensure that the number of distributed processes matches the allocated GPUs.

### 2. Model Architecture (`model_registry.py`)

The architectural configurations of DiagNO variants are defined in the `model_registry.py` module.

These configurations control model capacity and structure, including embedding dimensions, hidden feature dimensions, and hierarchical network settings.

To experiment with different model configurations, modify the corresponding definitions before training.

## Inference and Evaluation

The repository provides an inference pipeline for evaluating trained models through continuous autoregressive forecasting.

The inference process generates multi-step predictions and computes quantitative error metrics for evaluating forecasting performance.

### Running Inference

After completing model training and fine-tuning, run:

```bash
bash run_infer.sh
```

Before execution, verify that the script points to the correct model checkpoints, evaluation dataset, and output directory.

### Inference Configuration (`run_infer.sh`)

The following parameters can be customized:

- **`INFER_STEPS`**: Total number of autoregressive prediction steps (e.g., 10).
- **`SAVE_STEPS`**: A space-separated list of prediction steps at which output tensors are saved (e.g., `"5 10"`).
- **`NUM_EVAL_SAMPLES`**: Number of validation or test samples used for evaluation.

The evaluation pipeline supports multi-step forecasting and produces numerical results and saved prediction tensors for downstream analysis and visualization.

## Data Availability

The experiments involve both simulated spherical fluid dynamics and large-scale Earth system reanalysis datasets.

### Spherical Shallow Water Equations (SSWE)

The repository provides source code and utilities for generating and processing simulated SSWE trajectories.

Relevant implementations can be found in:

- `dataset/pde_dataset_ShallowWater.py`
- `dataset/pde_dataset_ShallowWater_steps.py`
- `dataset/solver/shallow_water_equations.py`

These utilities support the construction of synthetic fluid dynamics experiments on spherical grids.

The complete pre-generated training dataset is not included in the repository.

### ERA5 Atmospheric Dataset

ERA5 is a global atmospheric reanalysis dataset produced by the European Centre for Medium-Range Weather Forecasts (ECMWF).

Our atmospheric forecasting experiments use three variables at the 500 hPa pressure level:

- Geopotential (`Z`)
- Zonal wind (`U`)
- Meridional wind (`V`)

ERA5 data can be accessed through the official [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

### GLORYS12 Ocean Dataset

GLORYS12 is a global ocean reanalysis product providing high-resolution representations of ocean circulation and thermodynamic conditions.

Our ocean surface forecasting experiments involve four variables:

- Sea Surface Height (`SSH`)
- Sea Surface Temperature (`SST`)
- Zonal Velocity (`U`)
- Meridional Velocity (`V`)

GLORYS12 data can be obtained from the official [Copernicus Marine Service](https://data.marine.copernicus.eu/).

### Data Preparation

Due to the substantial storage requirements of ERA5 and GLORYS12, these reanalysis datasets are not hosted in this repository.

Users should download the required variables from the official data portals and prepare them according to the spatial resolutions, temporal sampling intervals, and experimental configurations described in the paper.

Dataset paths should be updated in the corresponding training and inference configurations before execution.

## Output Structure

The training and inference pipelines organize experimental outputs under the configured `ROOT_PATH`.

### Model Checkpoints

- `best_model_pretrain.pt`: Best checkpoint from the pretraining stage.
- `best_model_finetune.pt`: Best checkpoint from the fine-tuning stage.
- `latest_model.pt`: Most recently saved model checkpoint.

### Training Logs

- `history.csv`: Records epoch-level training losses and evaluation metrics.

### Inference Metrics

- `metrics_{model_name}_{steps}steps.csv`: Contains forecasting error metrics, including MAE, MSE, and RMSE, across autoregressive prediction steps.

### Saved Prediction Tensors

- `specific_steps_results_{steps}steps/`: Contains saved NumPy arrays for ground-truth (`tar`) and predicted (`prd`) fields at the steps specified by `SAVE_STEPS`.

These outputs can be used for downstream physical analysis, quantitative comparisons, and visualization.

## Code References

Our implementation integrates and adapts components from the following open-source projects:

- **Neural-Solver-Library**: https://github.com/thuml/Neural-Solver-Library
- **Torch-Harmonics**: https://github.com/NVIDIA/torch-harmonics

We acknowledge the authors and contributors of these projects for providing valuable open-source resources.

Any redistributed or adapted third-party components remain subject to their respective licenses and attribution requirements.

---

## Environment Setup

This section provides instructions for configuring the Python environment required to run DiagNO.

The reference environment uses **Python 3.10**, **PyTorch 2.4.0**, and **CUDA 12.1**, together with scientific computing, geospatial processing, and machine learning libraries.

### Prerequisites

Before installation, ensure that your system meets the following requirements:

- **Operating System**: Linux is recommended for running the provided Bash and distributed training scripts.
- **Package Manager**: Anaconda or Miniconda.
- **Python**: Version 3.10.
- **GPU**: NVIDIA GPU with compatible drivers.
- **CUDA**: NVIDIA driver compatible with CUDA 12.1-based PyTorch builds.

### Installation Steps

**Step 1: Create and Activate the Conda Environment**

Create a dedicated Python environment named `diagno_env`:

```bash
conda create -n diagno_env python=3.10 -y
conda activate diagno_env
```

**Step 2: Install Core Scientific and Geospatial Packages**

Install the fundamental libraries required for scientific computing, parallel processing, and geospatial data handling:

```bash
conda install -y \
    numpy scipy pandas matplotlib scikit-learn \
    h5py netcdf4 xarray dask \
    mpi4py cmake \
    cartopy shapely pyproj \
    pytz tqdm
```

**Step 3: Install PyTorch (CUDA 12.1)**

Install PyTorch 2.4.0 and its associated vision and audio packages using the official CUDA 12.1 wheel index:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Install PyTorch Geometric (PyG) Dependencies**

Install PyTorch Geometric:

```bash
pip install torch-geometric
```

For additional PyG operations, install the corresponding compiled extensions:

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

Please ensure that the selected extension wheels are compatible with your PyTorch version and CUDA environment.

**Step 5: Install Auxiliary Tools, Storage Libraries, and Data APIs**

Install additional machine learning utilities and experiment tracking tools:

```bash
pip install \
    aiohttp anyio attrs boto3 botocore timm \
    einops filelock fsspec huggingface-hub \
    jinja2 joblib networkx pillow protobuf \
    pydantic requests rich sympy tqdm typer wandb
```

Install 3D visualization and chunked data storage packages:

```bash
pip install vtk zarr numcodecs
```

Install auxiliary scientific packages and APIs for accessing climate and ocean datasets:

```bash
pip install astropy copernicusmarine cdsapi
```

### Environment Configuration File

The repository also includes an `environment.yml` file for reference.

Users can inspect this file to review the project dependency configuration and adapt the environment to their operating system, GPU hardware, and CUDA installation.

If the file defines a complete Conda environment, it can also be used as the basis for environment creation:

```bash
conda env create -f environment.yml
```

When using the environment file, check its environment name and package versions to avoid mixing incompatible installations.

---

## Contact

For any questions or discussions regarding this work, please contact:

**Herui Li**  
Shanghai Jiao Tong University  
Email: li-herui@sjtu.edu.cn

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{li2026diagno,
  title     = {{DiagNO}: Diagonal Spherical Neural Operators for Heterogeneous Earth Dynamics Modeling},
  author    = {Herui Li and Bin Lu and Haonan Qi and Lei Zhou and Luoyi Fu and Xinbing Wang and Meng Jin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
