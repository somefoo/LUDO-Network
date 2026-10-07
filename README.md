# LUDO-network

This repository trains and evaluates a deformable scene completion network from point-cloud inputs.
It supports training, evaluation, NRRD volume export, and single-sample inference with explainability exports.

To generate training and test datasets for this network, see [LUDO-Data](https://github.com/somefoo/LUDO-Data).

## Environment setup

Recommended OS: **Ubuntu 24.04**.

```bash
conda create --name deform python=3.11
conda activate deform
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install wandb torchmetrics matplotlib
```

## Repository layout

```text
.
├── main_pnpp.py                      # Entry point for train/test/nrrd/single modes
├── deformation_network.py            # Model definition
├── configs/
│   └── example/stanford_bunny.yaml   # Example standalone config
├── data_loaders/
│   ├── data_loader_geometric.py      # Dataset preprocessing + train/val/test split
│   └── read_pcd.py                   # PCD loading utilities
├── transforms/                       # Geometric augmentations + normalization
├── metrics/                          # Accuracy, IoU, and bounding-box evaluation
└── data_writers/                     # NRRD and PCD export writers
```

## Key concepts

| Term | Meaning |
| --- | --- |
| `image.pcd` | The observed input point cloud derived from a depth image. It is not an RGB image. |
| occupancy | Query points labeled as outside (`0`) or as an object segment class (`1..N`). |
| distance field | Auxiliary supervision for the closest-surface direction and signed distance. |
| processed cache | Cached `training.pt`, `validation.pt`, and `test.pt` files produced by the PyTorch Geometric dataset loader. |
| `run_type` | The selected runtime mode such as `train`, `test`, `nrrd`, or `single`. |

## Dataset location and file storage

At runtime, `main_pnpp.py` constructs the dataset path as:

```text
<dataset_prefix_path>/<scene>/
```

Where `dataset_prefix_path` and `scene` are read from YAML config.

Expected raw files:

```text
<dataset_prefix_path>/<scene>/
  raw_train/
    <id>_occupancy.pcd
    <id>_image.pcd
    <id>_bounding_boxes.csv
  raw_test/
    <id>_occupancy.pcd or <id>_occupancy.npz
    <id>_image.pcd
    <id>_bounding_boxes.csv
```

### Processed dataset cache (auto-generated)

PyTorch Geometric stores preprocessed tensors under:

```text
<dataset_prefix_path>/<scene>/processed/
  training.pt
  validation.pt
  test.pt
```

Notes:
- Train/validation are split automatically from `raw_train` (90%/10%).
- Test is built from `raw_test`.
- Set `force_reprocess: true` in config to regenerate cached processed files.

## Output locations (important)

### 1) Model checkpoints

Saved to:

```text
<log_model_path>/
```

Configured via `log_model_path` in YAML.

Files include:
- `model.pth` (final model)
- periodic backups: `model_000100.backup.pth`, etc. (frequency controlled by `log_model_period`)

### 2) NRRD exports

Written by `NRRDWriter` to:

```text
/tmp/nrrds/
  out_00000.nrrd
  out_00000.png
  ...
```

### 3) PCD inference/explainability exports

Written by `PCDWriter` and `PCDWriterEx` to:

```text
/tmp/pcds/
  *.pcd
```

### 4) Experiment tracking

`wandb` is enabled in `train` mode and disabled in non-training modes.

To disable Weights & Biases logging even during `train`, set `WANDB_MODE=disabled` before running.

```bash
# Fully disable wandb logging
export WANDB_MODE=disabled
```

Alternatively, you can disable it for one run:

```bash
WANDB_MODE=disabled python main_pnpp.py configs/example/stanford_bunny.yaml
```

## Running

Use `main_pnpp.py` with a config file:

```bash
python main_pnpp.py configs/example/stanford_bunny.yaml
```

Config loading behavior:
- The selected config file is loaded directly; there is no implicit base-config merge.
- Keys ending in `path` are converted to `Path` objects at load time.

## Config reference

### Core runtime

| Key | Type | Meaning |
| --- | --- | --- |
| `run_type` | string | Runtime mode. Supported values are `train`, `test`, `nrrd`, and `single`. |
| `dataset_prefix_path` | path | Base path that contains the dataset folder. |
| `scene` | string | Dataset folder name under `dataset_prefix_path`. |
| `log_model_path` | path | Directory used to store or load model checkpoints. |
| `force_reprocess` | boolean | If `true`, rebuild the processed PyG cache from the raw dataset. |

### Model and optimization

| Key | Type | Meaning |
| --- | --- | --- |
| `number_of_classes` | integer | Maximum number of occupancy classes the model can predict. |
| `positional_encoding_min_inclusive` | integer | Lower bound for the positional encoding frequency range. |
| `positional_encoding_max_inclusive` | integer | Upper bound for the positional encoding frequency range. |
| `hidden_layers_in_mlp` | integer | Number of hidden layers in the decoder MLP. |
| `dropout` | number | Dropout probability used inside the model. |
| `loss_type` | string | Loss combination. Supported values are `occ`, `sdf`, `occ+sdf`, and `occ+sdf+cos`. |
| `learning_rate` | number | Initial optimizer learning rate. |
| `learning_rate_decay` | boolean | Enables the learning-rate scheduler. |
| `clamp_distance` | number | Clamp value for signed-distance supervision. |
| `epochs` | integer | Number of training epochs. |
| `batch_size` | integer | Batch size for training and validation. |

### Logging, export, and augmentation

| Key | Type | Meaning |
| --- | --- | --- |
| `log_accuracy_period` | integer | Epoch interval for accuracy logging. |
| `log_iou_period` | integer | Epoch interval for IoU logging. |
| `log_model_period` | integer | Epoch interval for writing backup checkpoints. |
| `nrrd_resolution` | integer triplet | Resolution used in `nrrd` mode. |
| `nrrd_range` | integer pair | Inclusive test-sample index range used in `nrrd` mode. |
| `nrrd_resolution_test` | integer triplet | Resolution used for NRRD export in `test` mode. |
| `nrrd_resolution_train` | integer triplet | Resolution used for train-time NRRD preview generation. |
| `occupancy_pcd_resolution` | integer | Sampling resolution for occupancy PCD export in `single` mode. |
| `explainability_pcd_resolution` | integer | Sampling resolution for explainability export in `single` mode. |
| `explainability_radius_factor` | number | Optional radius scaling factor for explainability export in `single` mode. |
| `augment_rotation` | boolean | Enables random rotation augmentation during training. |
| `augment_rotation_degrees` | number | Rotation range used when `augment_rotation` is enabled. |
| `augment_mirror` | boolean | Enables axis mirroring augmentation during training. |
| `augment_maximum_drop_point_probability` | number in `[0, 1]` | Maximum point-drop probability for random point-drop data augmentation during training. |
| `architecture` | string | Informational config value used in the example config; the current code path uses the PointNet++ model defined in `deformation_network.py`. |

## Run modes (`run_type`)

- `train`: train and periodically log accuracy/IoU + NRRD previews.
- `test`: evaluate IoU, generate NRRD artifacts, run bounding-box test.
- `nrrd`: load a trained model and generate NRRD volumes.
- `single`: run single-sample inference and export occupancy + explainability PCDs.

## Inference variants (`run_type: single`)

Single-sample inference exports three occupancy variants:
- Without uncertainty: standard occupancy prediction.
- With activation-based uncertainty: entropy from one forward pass (small performance impact).
- With MC-dropout uncertainty: Monte Carlo dropout over multiple forward passes (big performance impact).

## Tips

- At inference time, keep the observed input point cloud small when possible. Fewer than `2000` points is often a good target for practical runtime.
- If the input cloud is much denser, randomly dropping points is often a reasonable first step and can still produce good results.
- If you need more runtime performance, increasing `augment_maximum_drop_point_probability` can help. The point-cloud encoder is often a practical bottleneck, so feeding it fewer observed points can noticeably reduce cost. Note that this setting is ultimately a data augmentation during training: for each sample, the actual drop probability is chosen randomly between `0` and the configured maximum.
- Camera motion during training data generation matters a lot. Restrict it to the range that is actually relevant for your application, and if you can assume something about the object pose relative to the camera, encode that in the generated training data instead of sampling unnecessarily broad viewpoints. This can substantially improve learning speed and output quality.

## Citation

If you use this network project, please cite LUDO:

```bibtex
@ARTICLE{henrich2025ludo,
  author={Henrich, Pit and Mathis-Ullrich, Franziska and Scheikl, Paul Maria},
  journal={IEEE Transactions on Robotics}, 
  title={LUDO: Low-Latency Understanding of Deformable Objects Using Point Cloud Occupancy Functions}, 
  year={2025},
  volume={41},
  number={},
  pages={4283-4299},
  doi={10.1109/TRO.2025.3582837}}
```
