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