# Prototypes of the Mind: A Unified Framework for Probing the Visual Brain

This is the official repository for "Prototypes of the Mind: A Unified Framework for Probing the Visual Brain" published in NeurIPS. The repository includes codebase for prototype leanring, fMRI mapping, and various analyses presented in the paper. 

## 1. Installation

Python **3.11+**, a compatible PyTorch installation, and a CUDA-capable GPU are recommended. Install a CUDA-compatible PyTorch build for your system first if necessary, then install the remaining packages:

```bash
# Run from the repository root, in your chosen Python environment.
python -m pip install -r requirements.txt
```

## 2. Configure paths

Training scripts accept paths through their **command-line flags**. Analysis notebooks read defaults from [`project_paths.py`](project_paths.py). Update that file or override its paths using environment variables (all optional):

```bash
# Example paths — change these to your local directories.
export PROTOMRI_DATA_ROOT=/path/to/your/data
export PROTOMRI_OUTPUT_ROOT=/path/to/your/runs
# Optional more specific overrides:
# export PROTOMRI_NSD_ROOT=/path/to/NSD_dataset
# export PROTOMRI_COCO_IMAGES=/path/to/mscoco/val2014
# export PROTOMRI_PROTOTYPE_RUN=/path/to/prototype/checkpoints
# export PROTOMRI_FMRI_RUNS=/path/to/fmri/checkpoints
# export PROTOMRI_FMRI_MULTI_RUNS=/path/to/multi_roi/checkpoints
```

Without overrides, notebooks use `./data` and `./runs` under the repository root. The examples below use those defaults. Run scripts and start Jupyter **from the repository root** so that paths stay consistent.

### Expected data layout

Feature extraction reads the original images from the locations below. The NSD images are the **preprocessed Algonauts Project 2023 Challenge** training images, not the raw NSD HDF5 images. The loader relies on their original filename convention (e.g., `train-0001_nsd-00013.png`), which is required to align images to the fMRI rows.

```text
ProtoMRI-main/
├── get_features.py                   # unified feature extraction
├── weights/
│   └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth  # user-provided
├── data/
│   ├── mscoco/
│   │   ├── train2014/COCO_train2014_*.jpg
│   │   └── val2014/COCO_val2014_*.jpg
│   ├── imagenet/                     # optional, for ImageNet-based experiments
│   │   ├── train/<class>/*.JPEG
│   │   └── val/<class>/*.JPEG         # validation reorganized by class
│   ├── NSD_dataset/                  # Algonauts 2023 Challenge format
│   │   └── subj01/
│   │       ├── training_split/
│   │       │   ├── training_images/train-0001_nsd-00013.png
│   │       │   └── training_fmri/{lh,rh}_training_fmri.npy
│   │       └── roi_masks/            # e.g. lh.floc-bodies_challenge_space.npy
│   └── features/                     # generated with get_features.py
│       ├── coco/COCO_train2014_*.npy  # flat folder, both train + val
│       ├── imagenet/<class>/*.npy     # no additional train/val directory
│       └── nsd/subj01/*.npy
└── runs/                             # training outputs
```

For generating files in the `features` folder, see the next step.

## 3. Extract DINOv3 patch features

Use **one script**—[`get_features.py`](get_features.py)—for all three datasets. The default backbone is `dinov3_vitb16`, and the default image resize is 256 × 256, producing 256 patch tokens per image. Download the DINOv3 checkpoint from [Meta's official model release](https://github.com/facebookresearch/dinov3#pretrained-models) following their access instructions. 

### 3.1 MS COCO 2014 (prototype training)

```bash
python get_features.py \
  --dataset coco \
  --input-dir ./data/mscoco \
  --output-dir ./data/features/coco \
  --weights ./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### 3.2 ImageNet (to train diffusion models for stimulus intervention)

```bash
python get_features.py \
  --dataset imagenet \
  --input-dir ./data/imagenet \
  --output-dir ./data/features/imagenet \
  --weights ./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### 3.3 NSD (Algonauts Project 2023 preprocessed images)

```bash
python get_features.py \
  --dataset nsd \
  --input-dir ./data/NSD_dataset \
  --output-dir ./data/features/nsd \
  --weights ./weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## 4. Learn and inspect prototypes

### 4.1 Train the prototype dictionary

To learn interpretable prototypes using the autoregressive objective, run the following command:

```bash
python main_prototype.py \
  --mode train \
  --feat_path ./data/features/coco \
  --log_dir ./runs/prototypes
```

By default, it uses hyperparameters specified in the paper.

### 4.2 Identify representative images for each prototype

To interprete the meanings of individual prototypes, we use the following command to visualize their representative images:

```bash
python main_prototype.py \
  --mode get_top_image \
  --feat_path ./data/features/coco \
  --image_path ./data/mscoco/val2014 \
  --weights ./runs/prototypes/best_model.pt \
  --log_dir ./runs/prototypes
```

The command will produces a pickle file storing data for visualization. Run the Jupyter notebook `visualize_top_image.ipynb` to generate images for visualization. By default, the notebook assumes **20 images per prototype**; update its grid layout if `--top_image` changes. 

### 4.3 Extract prototype activations for NSD

Before learning the prototypes-to-fMRI mapping, we need to first extract the prototype activations for NSD data:

```bash
python main_prototype.py \
  --mode get_nsd_proto \
  --feat_path ./data/features/nsd \
  --weights ./runs/prototypes/best_model.pt \
  --log_dir ./runs/prototypes
```

**Outputs:** `runs/prototypes/nsd_proto/subj01.pkl`, `subj02.pkl`, etc., each containing `image_name` and `prototype_similarity` (number of images × **64 spatial positions** × **1,000 prototypes** using the default settings).

## 5. Learn and evaluate the fMRI encoding model

To learn an independent encoding model for each brain ROI, run the following command:

```bash
python main_fmri.py \
  --mode train \
  --proto_dir ./runs/prototypes/nsd_proto \
  --mri_dir ./data/NSD_dataset \
  --subject subj01 \
  --area_name bodies \
  --patch_name EBA \
  --log_dir ./runs/fmri/subj01/EBA
```

Please adjust `patch_name` (ROI name) and `area_name` (functional selectivity defined in NSD) accordingly.

### 5.1 Evaluate a single-ROI checkpoint

Entrypoint: `eval()` in `main_fmri.py`, selected by `--mode eval`.

```bash
python main_fmri.py \
  --mode eval \
  --proto_dir ./runs/prototypes/nsd_proto \
  --mri_dir ./data/NSD_dataset \
  --subject subj01 \
  --area_name bodies \
  --patch_name EBA \
  --weights ./runs/fmri/subj01/EBA/best_model.pt
```

The evaluation prints the **mean voxel-wise Pearson correlation** across images (the function `voxelwise_pearsonr`). Both the single-ROI (`NSD_proto`) and multi-ROI (`NSD_proto_multiple`) datasets now use **independent 80% train / 10% validation / 10% test partitions**, generated reproducibly using `--seed` (default `88`). 

### 5.2 Fit one model across multiple ROIs

To train a single model for multiple instead of a single ROI, edit `main()` in `main_fmri.py`:

1. **Comment out** the two default `NSD_proto(...)` assignments for `train_dataset` and `val_dataset`.
2. **Uncomment** the two `NSD_proto_multiple(...)` assignments immediately below them. The included example combines `['faces', 'bodies', 'places']` with `['OFA', 'EBA', 'OPA']`.
3. Run the same training command with an explicit hemisphere (for example `--hemi rh`) and a **separate** `--log_dir`, such as `./runs/fmri_multi/subj01/rh`.


## 6. Analysis notebooks

Launch Jupyter from this repository root and edit the notebook configuration cells (or `project_paths.py`) as needed.

| Analysis | Entrypoint | What it does |
| --- | --- | --- |
| **Brain-area selectivity** | [`get_top_proto.ipynb`](get_top_proto.ipynb) | Ranks prototypes by the mean voxel readout weight of a trained ROI encoding model; Note that the code assumes the representative images for prototypes are already generated. |
| **Mapping prototypes to multiple ROIs** | [`proto_map_visualization.ipynb`](proto_map_visualization.ipynb) | Uses a **multi-ROI** checkpoint and ROI masks to project each prototype's voxel readout weights onto a brain surface. The final cell iterates over **every prototype**, one at a time. |
| **Spatial tuning of a brain area** | [`visualize_attention.ipynb`](visualize_attention.ipynb) | Overlays ROI-specific, image-aligned spatial activation maps onto NSD images. |
| **Stimulus intervention** | **TODO (include the modified JiT code)** | |


## 8. Reference
If you use our code or data, please cite our paper:
```
@InProceedings{proto_mri,
author = {Chen, Shi and Ahn, Seoyoung and Tsao, Doris},
title = {Prototypes of the Mind: A Unified Framework for Probing the Visual Brain},
booktitle = {NeurIPS},
year = {2026}
}
```
