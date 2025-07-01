import argparse
import os
from easyDataverse import Dataverse

dataset_dict = {'tension': {'spect': 'doi:10.7910/DVN/YLQGUO',
                            'vol': 'doi:10.7910/DVN/G5DLI7',
                            'star': 'doi:10.7910/DVN/9URYI1'},
                'shear': {'spect': 'doi:10.7910/DVN/KZDRUE',
                            'vol': 'doi:10.7910/DVN/OCVQJ1',
                            'star': 'doi:10.7910/DVN/APUKE5'}
}

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download dataset based on case and energy decomposition')
    parser.add_argument('--case', type=str, required=True, choices=['tension', 'shear'],
                       help='Case type: tension or shear')
    parser.add_argument('--decomp', type=str, required=True, choices=['spect', 'vol', 'star'],
                       help='Energy decomposition type: spect, vol, or star')
    
    args = parser.parse_args()
    
    # Get the DOI from the dataset dictionary
    if args.case in dataset_dict and args.decomp in dataset_dict[args.case]:
        doi = dataset_dict[args.case][args.decomp]
    else:
        print(f"Error: Invalid combination of case '{args.case}' and decomposition '{args.decomp}'")
        return
    
    # Create output directory
    outdir = os.path.join("data", f"{args.case}/{args.decomp}")
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Downloading dataset for case: {args.case}, decomposition: {args.decomp}")
    print(f"DOI: {doi}")
    print(f"Output directory: {outdir}")
    
    # Initialize dataverse and download dataset
    dataverse = Dataverse("https://dataverse.harvard.edu")
    dataset = dataverse.load_dataset(
        pid=doi,
        filedir=outdir,
    )
    print(f"Dataset downloaded successfully to {outdir}")
    print(dataset)

if __name__ == "__main__":
    main()