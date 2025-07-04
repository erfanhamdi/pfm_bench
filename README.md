# Dataset Download Script

This script downloads datasets from Harvard Dataverse based on case type and energy decomposition method.

## Prerequisites

Make sure you have the required Python package installed:

```bash
pip install easyDataverse
```

## Usage

The script accepts two required command-line arguments:

### Arguments

- `--case`: The case type (required)
  - Options: `tension` or `shear`
- `--decomp`: The energy decomposition method (required)
  - Options: `spect`, `vol`, or `star`

### Examples

```bash
# Download tension spectral data
python download_data.py --case tension --decomp spect

# Download shear volume data
python download_data.py --case shear --decomp vol

# Download tension star decomposition data
python download_data.py --case tension --decomp star

# Download shear spectral data
python download_data.py --case shear --decomp spect
```

## Available Datasets

The script supports the following dataset combinations:

| Case | Decomposition | DOI |
|------|---------------|-----|
| tension | spect | doi:10.7910/DVN/YLQGUO |
| tension | vol | doi:10.7910/DVN/G5DLI7 |
| tension | star | doi:10.7910/DVN/9URYI1 |
| shear | spect | doi:10.7910/DVN/KZDRUE |
| shear | vol | doi:10.7910/DVN/OCVQJ1 |
| shear | star | doi:10.7910/DVN/APUKE5 |

## Output Structure

The script creates a structured folder hierarchy in the `data` directory:

```
data/
├── tension/
|  ├── spect/
|  ├── vol/
|  └── star/
├── shear/
|  ├── spect/
|  ├── vol/
|  └── star/
```

Each folder contains the downloaded dataset files for the corresponding case and decomposition method.

## Error Handling

The script includes validation for:
- Valid case and decomposition combinations
- Required argument presence
- Automatic directory creation

If an invalid combination is provided, the script will display an error message and exit gracefully.

## UNet Model

This repository also contains an implementation of a 2-D UNet for segmenting the time–evolution data that you download with `download_data.py`.
The core code lives in `src/models/unet/`:

* `unet.py` – network architecture and custom loss functions (Dice, Focal, Combined)
* `utils.py` – `torch.utils.data.Dataset` wrapper that reads the HDF5 files produced by the download script
* `train.py` – full training loop with validation and [Weights & Biases](https://wandb.ai/) logging

### Additional Prerequisites

Besides the `easyDataverse` package mentioned above, UNet training requires the following Python packages:

```bash
pip install torch torchvision matplotlib h5py wandb numpy
```

### Training the Model

After you have downloaded a dataset you can start training right away.  The minimal command is

```bash
python src/models/unet/train.py --data_dir <path-to-dataset>
```

The script exposes many hyper-parameters; the most important ones are shown below (defaults in parentheses):

```bash
--epochs        Number of training epochs               (100)
--batch_size    Batch size                              (16)
--learning_rate Initial learning rate                   (1e-4)
--in_channels   Number of input/output channels         (1)
--alpha         Weight for Dice loss in CombinedLoss    (0.5)
--beta          Weight for Focal loss in CombinedLoss   (0.5)
--threshold     Threshold used to binarise GT masks     (0.4)
```

Example: train a UNet on the tension/spectral dataset with a smaller batch size for GPU-memory reasons:

```bash
python src/models/unet/train.py \
  --data_dir data/tension/spect \
  --epochs 200 \
  --batch_size 8 \
  --learning_rate 5e-5
```

During training the script will automatically

1. create `src/models/unet/results/` where checkpoints are stored,
2. log metrics and example predictions to your **wandb** project (set by the script – internet optional), and
3. save the best and final model weights as `*.pt` files.