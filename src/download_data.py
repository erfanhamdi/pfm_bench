import argparse
import os
from easyDataverse import Dataverse

# Mapping of available datasets and their DOIs
DATASET_DOIS = {
    "tension": {
        "spect": "doi:10.7910/DVN/G3QRE0",
        "vol": "doi:10.7910/DVN/G5DLI7",
        "star": "doi:10.7910/DVN/9URYI1",
    },
    "shear": {
        "spect": "doi:10.7910/DVN/KZDRUE",
        "vol": "doi:10.7910/DVN/OCVQJ1",
        "star": "doi:10.7910/DVN/APUKE5",
    },
    "test": {
        "spect": "doi:10.7910/DVN/G3QRE0",
    },
}


def download_dataset(case: str, decomp: str, outdir: str | None = None):
    """Download a dataset from Harvard Dataverse.

    Parameters
    ----------
    case : {"tension", "shear"}
        Loading case.
    decomp : {"spect", "vol", "star"}
        Energy decomposition type.
    outdir : str | None, optional
        Destination directory (defaults to ``data/{case}/{decomp}``).
    """
    if case not in DATASET_DOIS:
        raise ValueError(f"Unknown case '{case}'. Choose from {list(DATASET_DOIS.keys())}.")
    if decomp not in DATASET_DOIS[case]:
        raise ValueError(
            f"Unknown decomposition '{decomp}' for case '{case}'. Choose from {list(DATASET_DOIS[case].keys())}."
        )

    doi = DATASET_DOIS[case][decomp]
    outdir = outdir or os.path.join("data", case, decomp)
    os.makedirs(outdir, exist_ok=True)

    print(f"Downloading dataset for case: {case}, decomposition: {decomp}")
    print(f"DOI: {doi}")
    print(f"Output directory: {outdir}")

    dataverse = Dataverse("https://dataverse.harvard.edu")
    dataset = dataverse.load_dataset(pid=doi, filedir=outdir)

    print(f"Dataset downloaded successfully to {outdir}")
    return dataset


def main():  # pragma: no cover – CLI entry point
    """Entry point for the ``download-data`` CLI script."""
    parser = argparse.ArgumentParser(
        description="Download dataset based on case and energy decomposition",
    )
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        choices=["tension", "shear"],
        help="Case type: tension or shear",
    )
    parser.add_argument(
        "--decomp",
        type=str,
        required=True,
        choices=["spect", "vol", "star"],
        help="Energy decomposition type: spect, vol, or star",
    )

    args = parser.parse_args()
    download_dataset(case=args.case, decomp=args.decomp)


if __name__ == "__main__":
    main() 