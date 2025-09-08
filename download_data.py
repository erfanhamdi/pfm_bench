import argparse
import os
from easyDataverse import Dataverse
import re
dataset_dict = {'tension': {'spect': 'doi:10.7910/DVN/YLQGUO',
                            'vol': 'doi:10.7910/DVN/G5DLI7',
                            'star': 'doi:10.7910/DVN/9URYI1'},
                'shear': {'spect': 'doi:10.7910/DVN/KZDRUE',
                            'vol': 'doi:10.7910/DVN/OCVQJ1',
                            'star': 'doi:10.7910/DVN/APUKE5'}
}
def download_individually(doi, outdir):
    import os
    import requests
    from urllib.parse import unquote

    dataset_api_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={doi}"
    os.makedirs(outdir, exist_ok=True)

    print("🔍 Fetching dataset metadata...")
    response = requests.get(dataset_api_url)
    response.raise_for_status()
    data = response.json()

    files = data["data"]["latestVersion"]["files"]
    print(f"📁 Found {len(files)} files.")

    # Step 3: Download each file
    for f in files:
        file_id = f["dataFile"]["id"]
        orig_name = f["dataFile"]["filename"]
        size = f["dataFile"].get("filesize", "unknown")
        
        download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        output_path = os.path.join(outdir, orig_name)

        if os.path.exists(output_path) and size != "unknown" and os.path.getsize(output_path) == size:
            print(f"✅ Already exists with same size: {orig_name}")
            continue

        print(f"⬇️ Downloading: {orig_name} ({size} bytes)")
        

        try:
            r = requests.get(download_url, stream=True)
            r.raise_for_status()

            with open(output_path, "wb") as out_file:
                for chunk in r.iter_content(chunk_size=8192):
                    out_file.write(chunk)

            print(f"✅ Saved: {output_path}")
        except Exception as e:
            print(f"❌ Failed to download {orig_name}: {e}")

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

def curl_download(doi, outdir):
    URL = f"https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId={doi}"
    curl_cmd = f'curl -L -O -J -k "{URL}"'
    print("Running the curl command...")
    cmd = f'cd "{outdir}" && {curl_cmd}'
    os.system(cmd)
    zip_file = os.path.join(outdir, "dataverse_files.zip")
    
    env = os.environ.copy()
    env['UNZIP_DISABLE_ZIPBOMB_DETECTION'] = 'TRUE'
    print("Extracting the dataset...")
    import subprocess
    unzip_result = subprocess.run(
        f"unzip {zip_file} -d '{outdir}'",
        shell=True,
        env=env,
        capture_output=False,
        text=True
    )
    
    if unzip_result.returncode == 0:
        os.remove(f"{outdir}/dataverse_files.zip")
        print("Unzip successful, original zip file removed.")
    else:
        print("Unzip failed, keeping original zip file.")
        print(f"Error: {unzip_result.stderr}")
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download dataset based on case and energy decomposition')
    parser.add_argument('--case', type=str, required=True, choices=['tension', 'shear'],
                       help='Case type: tension or shear')
    parser.add_argument('--decomp', type=str, required=True, choices=['spect', 'vol', 'star'],
                       help='Energy decomposition type: spect, vol, or star')
    parser.add_argument('--method', type=str, default='indie', choices=['ez', 'curl', 'indie'],
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
    if args.method == 'indie':
        download_individually(doi, outdir)
    print("Renaming files to have 8-digit zero-padded names")
    pad_filenames(outdir)
    print("File renaming completed.")
if __name__ == "__main__":
    main()