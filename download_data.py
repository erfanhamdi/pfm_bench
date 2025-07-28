import argparse
import os
from easyDataverse import Dataverse

dataset_dict = {'tension': {'spect': 'doi:10.7910/DVN/G3QRE0',
                            'vol': 'doi:10.7910/DVN/G5DLI7',
                            'star': 'doi:10.7910/DVN/9URYI1'},
                'shear': {'spect': 'doi:10.7910/DVN/KZDRUE',
                            'vol': 'doi:10.7910/DVN/OCVQJ1',
                            'star': 'doi:10.7910/DVN/APUKE5'}
}

def curl_download(doi, outdir):
    URL = f"https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId={doi}"
    curl_cmd = f'curl -L -O -J -k "{URL}"'
    print("Running the curl command...")
    cmd = f'cd "{outdir}" && {curl_cmd}'
    os.system(cmd)
    unzip_result = os.system("unzip dataverse_files.zip")
    if unzip_result == 0:
        os.remove(f"{outdir}/dataverse_files.zip")
    else:
        print("Unzip failed keeping the original files")
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download dataset based on case and energy decomposition')
    parser.add_argument('--case', type=str, required=True, choices=['tension', 'shear'],
                       help='Case type: tension or shear')
    parser.add_argument('--decomp', type=str, required=True, choices=['spect', 'vol', 'star'],
                       help='Energy decomposition type: spect, vol, or star')
    parser.add_argument('--method', type=str, required=True, choices=['ez', 'curl'],
                       help='Data Download Method')                       
    
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
    
    if args.method == 'ez':
        try:
            # Initialize dataverse and download dataset
            dataverse = Dataverse("https://dataverse.harvard.edu")
            dataset = dataverse.load_dataset(
                pid=doi,
                filedir=outdir,
            )
            print(f"Dataset downloaded successfully to {outdir}")
            print(dataset)
        except:
            curl_download(doi, outdir)
    if args.method == 'curl':
        curl_download(doi, outdir)

if __name__ == "__main__":
    main()