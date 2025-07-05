import torch
from unet import UNet
from utils import UNetDataset
from metrics.Dice import compute_dice
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

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
    with torch.no_grad():
        for batch_idx, (x, y, seed) in enumerate(data_loader):
            output_shape = [0, 2, 3, 4, 1]
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_dict['init'][seed[0]] = x.unsqueeze(-1).permute(output_shape)  #[b, h, w, t, c]
            pred_dict['gt'][seed[0]] = y.unsqueeze(-1).permute(output_shape)  #[b, h, w, t, c]
            pred_dict['pred'][seed[0]] = pred.unsqueeze(-1).permute(output_shape)  #[b, h, w, t, c]
    return pred_dict

# TODO: only for testing
def plot_pred(pred_dict, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    seed_keys = list(pred_dict['init'].keys())
    axes[0, 0].imshow(pred_dict['init'][seed_keys[0]][0, :, :, 0, 0])
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

def main(seeds, data_dir, model_path, out_dir, ds_size=-1, model_config=None, device='cpu'):
    test_dataset = UNetDataset(datadir=data_dir, split='test', num_c=model_config['num_channels'], train_ratio=0.0, val_ratio=0.0, test_ratio=1.0, ds_size=ds_size, return_seed=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for seed in seeds:
        case_name = data_dir.split('/')[-2]
        decomp_name = data_dir.split('/')[-1]
        model = load_unet_model_state(model_path, **model_config, device=device)
        pred_dict = evaluate(model, test_loader, device)
        os.makedirs(out_dir, exist_ok=True)
        pred_dict_dir = f"{out_dir}/preds_UNet_{case_name}_{decomp_name}_{seed}.pkl"
        with open(pred_dict_dir, "wb") as f:
            pickle.dump(pred_dict, f)
        print(f"Predictions saved to {pred_dict_dir}")
        compute_dice(pred_dict_dir, out_dir=out_dir, threshold_pred=0.5, threshold_gt=0.5)
        print(f"Dice scores saved to {out_dir}/dice_scores.pkl")
        plot_pred(pred_dict, out_dir)
        print(f"Predictions plot saved to {out_dir}")

if __name__ == "__main__":
    # TODO: the models should be saved in UNet/results/wandb_project/models
    # TODO: the preds should be saved in UNet/results/wandb_project/preds
    seeds = [3]
    data_dir = "data/tension/spect"
    model_path = "/projectnb/lejlab2/erfan/PF_Bench/unet/result/best_models/unet-128x128-0.4-dice-focal-05/aug_unet-1-dice-focal-05-128x128_best_53_val_loss_0.176743.pt"
    out_dir = "src/models/UNet/results/test_gh/preds"
    model_config = {
        "num_channels": 1,
        "channels": 32,
    }
    main(seeds, data_dir, model_path, out_dir, ds_size=10, model_config=model_config, device='cpu')