import argparse
import os
import re
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


def pad_filenames(directory: str):
    """Rename files with numeric names to have 8-digit zero-padded names.
    
    Parameters
    ----------
    directory : str
        Directory containing files to rename.
    """
    if not os.path.exists(directory):
        return
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
            
        # Extract the name and extension
        name, ext = os.path.splitext(filename)
        
        # Check if the name is purely numeric
        if name.isdigit():
            # Pad with zeros to make it 8 digits
            new_name = name.zfill(8)
            new_filename = new_name + ext
            new_filepath = os.path.join(directory, new_filename)
            
            if filename != new_filename:  # Only rename if different
                print(f"Renaming: {filename} -> {new_filename}")
                os.rename(filepath, new_filepath)
        
        # Also handle names that start with numbers (e.g., "123_data.txt" -> "00000123_data.txt")
        elif re.match(r'^\d+', name):
            # Extract the leading digits
            match = re.match(r'^(\d+)(.*)$', name)
            if match:
                number_part = match.group(1)
                rest_part = match.group(2)
                
                # Pad the number part to 8 digits
                padded_number = number_part.zfill(8)
                new_name = padded_number + rest_part
                new_filename = new_name + ext
                new_filepath = os.path.join(directory, new_filename)
                
                if filename != new_filename:  # Only rename if different
                    print(f"Renaming: {filename} -> {new_filename}")
                    os.rename(filepath, new_filepath)


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
    
    # Pad filenames with zeros to have 8-digit names
    print("Renaming files to have 8-digit zero-padded names")
    pad_filenames(outdir)
    print("File renaming completed.")
    
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