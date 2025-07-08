import torch
from unet import UNet
from utils import UNetDataset
from metrics.Dice import compute_dice_from_pred_dict
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import glob
import h5py
from tqdm import tqdm

def load_unet_model_state(model_path, num_channels=1, channels=32, device='cpu'):
    # TODO: change the name of inputs to the unet model
    model = UNet(in_c=num_channels, c=channels).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict_to_load = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_to_load)
    except Exception as e: print(f"Error loading model state_dict from {model_path}: {e}"); raise e
    return model

def evaluate(model, data_loader, device):
    model.eval()
    pred_dict = {"init":{}, "gt": {}, "pred": {}}
    
    # Get total number of batches for progress bar
    total_batches = len(data_loader)
    print(f"Total batches to process: {total_batches}")
    
    with torch.no_grad():
        for batch_idx, (x, y, seed) in tqdm(enumerate(data_loader),
                                           desc="Processing batches",
                                           total=total_batches):
            output_shape = [0, 2, 3, 1]
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred = torch.sigmoid(pred)
            pred_dict['init'][seed[0]] = x.permute(output_shape)  #[b, h, w, c]
            pred_dict['gt'][seed[0]] = y.permute(output_shape)  #[b, h, w, c]
            pred_dict['pred'][seed[0]] = pred.permute(output_shape)  #[b, h, w, c]
    return pred_dict

# TODO: only for testing
def plot_pred(pred_dict, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    seed_keys = list(pred_dict['init'].keys())
    axes[0, 0].imshow(pred_dict['init'][0, :, :, 0, 0])
    axes[1, 0].imshow(pred_dict['gt'][seed_keys[0]][0, :, :, 0, 0])
    axes[2, 0].imshow(pred_dict['pred'][seed_keys[0]][0, :, :, 0, 0])
    axes[0, 0].set_title(f"seed {seed_keys[0]}")
    axes[0, 1].imshow(pred_dict['init'][seed_keys[1]][0, :, :, 0, 0])
    axes[1, 1].imshow(pred_dict['gt'][seed_keys[1]][0, :, :, 0, 0])
    axes[2, 1].imshow(pred_dict['pred'][seed_keys[1]][0, :, :, 0, 0])
    axes[0, 1].set_title(f"seed {seed_keys[1]}")
    axes[1, 1].set_title(f"seed {seed_keys[1]}")
    axes[2, 1].set_title(f"seed {seed_keys[1]}")
    plt.savefig(f"{out_dir}/plot_pred.png")
    plt.close()

def main(data_dir, model_path, out_dir, ds_size=-1, model_config=None, threshold_pred=0.5, threshold_gt=0.5, device='cpu', seed=1):
    test_dataset = UNetDataset(datadir=data_dir, split='test', num_c=model_config['num_channels'], train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, ds_size=ds_size, return_seed=True, apply_transforms=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")
    case_name = data_dir.split('/')[-2]
    decomp_name = data_dir.split('/')[-1]
    model = load_unet_model_state(model_path, **model_config, device=device)
    pred_dict = evaluate(model, test_loader, device)
    os.makedirs(out_dir, exist_ok=True)
    pred_dict_dir = f"{out_dir}/preds_UNet_{case_name}_{decomp_name}_{seed}.hdf5"
    
    # Convert predictions to lists and save as HDF5
    pred_list = []
    gt_list = []
    init_list = []
    pred_hdf5 = h5py.File(pred_dict_dir, 'w')
    for key in pred_dict['pred'].keys():
        pred_list.append(pred_dict['pred'][key].squeeze(0))
        gt_list.append(pred_dict['gt'][key].squeeze(0))
        init_list.append(pred_dict['init'][key].squeeze(0))
    pred_hdf5.create_dataset('pred', data=torch.stack(pred_list), compression='gzip', compression_opts=9, shuffle=True)
    pred_hdf5.create_dataset('gt', data=torch.stack(gt_list), compression='gzip', compression_opts=9, shuffle=True)
    pred_hdf5.create_dataset('init', data=torch.stack(init_list), compression='gzip', compression_opts=9, shuffle=True)
    pred_hdf5.create_dataset('keys', data=list(pred_dict['pred'].keys()), compression='gzip', compression_opts=9, shuffle=True)
    pred_hdf5.close()
    
    print(f"Predictions saved to {pred_dict_dir}")
    compute_dice_from_pred_dict(pred_dict_dir, out_dir=out_dir, threshold_pred=threshold_pred, threshold_gt=threshold_gt)
    print(f"Dice scores saved to {out_dir}/dice_scores.pkl")
    # plot_pred(pred_dict, out_dir)
    # print(f"Predictions plot saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate UNet model')
    parser.add_argument('--data_dir', type=str, default='data/tension/spect', help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, default='', help='Path to the trained model checkpoint')
    parser.add_argument('--out_dir', type=str, default='', help='Output directory for predictions')
    parser.add_argument('--ds_size', type=int, default=-1, help='Dataset size limit (-1 for all)')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--channels', type=int, default=32, help='Number of channels in UNet')
    parser.add_argument('--threshold_pred', type=float, default=0.5, help='Threshold for predictions')
    parser.add_argument('--threshold_gt', type=float, default=0.5, help='Threshold for ground truth')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--wandb_project', type=str, default='test-UNet-Training', help='Wandb project name')
    parser.add_argument('--epoch', type=int, default=None, help='Specific epoch to load (latest if not specified)')
    
    args = parser.parse_args()
    
    # Construct paths based on directory structure if not provided
    case_name = args.data_dir.split('/')[-2]
    decomp_name = args.data_dir.split('/')[-1]
    
    if not args.model_path:
        models_dir = f"src/models/UNet/results/{args.wandb_project}/{case_name}_{decomp_name}/seed_{args.seed}/models"
        if args.epoch is not None:
            model_filename = f"UNet_epoch_{args.epoch}.pt"
            args.model_path = f"{models_dir}/{model_filename}"
        else:
            # Try to find final model first, then latest checkpoint
            final_model_path = f"{models_dir}/UNet_final.pt"
            if os.path.exists(final_model_path):
                args.model_path = final_model_path
                print(f"Using final model: {final_model_path}")
            else:
                # Find latest checkpoint
                checkpoint_pattern = f"{models_dir}/UNet_epoch_*.pt"
                checkpoint_files = glob.glob(checkpoint_pattern)
                if checkpoint_files:
                    # Extract epoch numbers and find the latest
                    epochs = []
                    for file in checkpoint_files:
                        try:
                            epoch_str = file.split('_')[-1].split('.')[0]
                            epochs.append((int(epoch_str), file))
                        except ValueError:
                            continue
                    if epochs:
                        latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
                        args.model_path = latest_file
                        print(f"Using latest checkpoint from epoch {latest_epoch}: {latest_file}")
                    else:
                        raise FileNotFoundError(f"No valid checkpoints found in {models_dir}")
                else:
                    raise FileNotFoundError(f"No checkpoints found in {models_dir}")
    
    if not args.out_dir:
        args.out_dir = f"src/models/UNet/results/{args.wandb_project}/{case_name}_{decomp_name}/seed_{args.seed}/preds"
    
    # Create model config from arguments
    model_config = {
        "num_channels": args.num_channels,
        "channels": args.channels,
    }
    
    main(
        data_dir=args.data_dir,
        model_path=args.model_path,
        out_dir=args.out_dir,
        ds_size=args.ds_size,
        model_config=model_config,
        threshold_pred=args.threshold_pred,
        threshold_gt=args.threshold_gt,
        device=args.device,
        seed=args.seed
    )