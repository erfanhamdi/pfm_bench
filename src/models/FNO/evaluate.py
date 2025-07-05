import torch
from fno import FNO2d
from utils import FNODataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_fno_model_state(model_path, num_channels=3, modes=12, width=20, initial_step=10, device='cpu'):
    """Load a trained FNO model from a state dict path."""
    model = FNO2d(num_channels=num_channels, modes1=modes, modes2=modes, width=width, initial_step=initial_step).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict_to_load = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_to_load)
    except Exception as e: print(f"Error loading model state_dict from {model_path}: {e}"); raise e
    return model

def evaluate(model, data_loader, initial_step, rollout_end_step, device):
    model.eval()
    pred_dict = {}
    with torch.no_grad():
        for batch_idx, (x, y, grid, seed) in enumerate(data_loader):
            x, y, grid = x.to(device), y.to(device), grid.to(device)
            inp_shape = list(x.shape)[:-2] + [-1]
            pred = y[..., :initial_step, :]
            for t in range(initial_step, rollout_end_step + 1):
                inp = x.reshape(inp_shape)
                im = model(inp, grid)
                pred = torch.cat((pred, im), -2)
                x = torch.cat((x[..., 1:, :], im), dim=-2)
            pred_dict[seed] = pred
    return pred_dict

# TODO: only for testing
def plot_pred(pred_dict):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    seed_keys = list(pred_dict.keys())
    print(seed_keys)
    axes[0, 0].imshow(pred_dict[seed_keys[0]][0, :, :, 0, 0])
    axes[0, 1].imshow(pred_dict[seed_keys[0]][0, :, :, 50, 0])
    axes[0, 2].imshow(pred_dict[seed_keys[0]][0, :, :, 100, 0])
    axes[0, 0].set_title(f"seed {seed_keys[0]}")
    axes[1, 0].imshow(pred_dict[seed_keys[1]][0, :, :, 0, 0])
    axes[1, 1].imshow(pred_dict[seed_keys[1]][0, :, :, 50, 0])
    axes[1, 2].imshow(pred_dict[seed_keys[1]][0, :, :, 100, 0])
    axes[1, 0].set_title(f"seed {seed_keys[1]}")
    plt.savefig("src/models/FNO/results/preds/FNO_tension_miehe_c64x64_3_300.png")
    plt.close()

def main(seeds, data_dir, results_path, ds_size=-1, rollout_end_step=100, model_config=None, device='cpu'):
    test_dataset = FNODataset(datadir=data_dir, split='test', num_c=model_config['num_channels'], train_ratio=0.0, val_ratio=0.0, test_ratio=1.0, ds_size=ds_size, return_seed=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for seed in seeds:
        case_name = data_dir.split('/')[-2]
        decomp_name = data_dir.split('/')[-1]
        # TODO: change this to the correct path
        # model_path = f"{results_path}/FNO_{case_name}_{decomp_name}_{seed}.pth"
        model_path = results_path
        model = load_fno_model_state(model_path, **model_config, device=device)
        pred_dict = evaluate(model, test_loader, model_config['initial_step'], rollout_end_step, device)
        with open(f"src/models/FNO/results/preds/FNO_{case_name}_{decomp_name}_{seed}.pkl", "wb") as f:
            pickle.dump(pred_dict, f)
        plot_pred(pred_dict)

if __name__ == "__main__":
    seeds = [3]
    data_dir = "data/tension/spect"
    results_path = "/projectnb/lejlab2/erfan/PF_Bench/FNO/models/FNO-normalized-128-refactor/FNO_tension_miehe_c64x64_3_300.pt"
    model_config = {
        "num_channels": 3,
        "modes": 12,
        "width": 20,
        "initial_step": 10,
    }
    main(seeds, data_dir, results_path, ds_size=10, rollout_end_step=100, model_config=model_config, device='cpu')