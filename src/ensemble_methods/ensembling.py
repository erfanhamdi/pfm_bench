import numpy as np
import torch
from metrics.Dice import compute_dice_from_pred_tensor
from torch.utils.data import random_split, DataLoader
from ensemble_methods.utils import EnsembleDataset, load_predictions
from ensemble_methods.stacking import Stacker
import argparse
import yaml
from ensemble_methods.stacking import train_stacking

def hard_voting(ensemble_loader, vote_cutoff=3, threshold_pred=0.5):
    for keys, pred_array, gt_array in ensemble_loader:
        binary_preds = (pred_array > threshold_pred).float()
        ensemble_pred = (binary_preds.sum(axis=1) >= vote_cutoff).float()
    return ensemble_pred, gt_array

def soft_voting(ensemble_loader):
    for keys, pred_array, gt_array in ensemble_loader:
        ensemble_pred = torch.mean(pred_array, axis=1)
    return ensemble_pred, gt_array

def stacking(ensemble_loader, model=None, model_address=None):
    if model is None:
        model = Stacker()
        model.load_state_dict(torch.load(model_address))
    # model.eval()
    with torch.no_grad():
        for keys, pred_array, gt_array in ensemble_loader:
            ensemble_pred = model(pred_array)
            ensemble_pred = torch.sigmoid(ensemble_pred)
    return ensemble_pred, gt_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UNet', choices=['FNO', 'UNet'])
    parser.add_argument('--config', type=str, default='src/ensemble_methods/ensembling_config.yml')
    args = parser.parse_args()
    print(f"Using config: {args.config}")
    print(f"Ensembling the predictions of: {args.model}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seed_pred_list = config['seed_pred_list']
    hard_voting_config = config['hard_voting']
    soft_voting_config = config['soft_voting']
    stacking_config = config['stacking']
    
    if args.model == 'FNO':
        seed_pred_list = seed_pred_list['FNO']
    elif args.model == 'UNet':
        seed_pred_list = seed_pred_list['UNet']
    
    ensemble_dataset = load_predictions(seed_pred_list)
    keys, pred_array, gt_array = ensemble_dataset
    ensemble_dataset = EnsembleDataset(pred_array, gt_array, keys)
    torch.manual_seed(1)
    np.random.seed(1)
    train_size = int(len(ensemble_dataset) * 0.6)
    val_size = int(len(ensemble_dataset) * 0.2)
    test_size = int(len(ensemble_dataset) * 0.2)
    meta_train_ds, meta_val_ds, meta_test_ds = random_split(ensemble_dataset, [train_size, val_size, test_size])
    meta_test_loader = DataLoader(meta_test_ds, batch_size=test_size, shuffle=False)
    
    # hard voting
    ensemble_preds, ensemble_gt = hard_voting(meta_test_loader, vote_cutoff=hard_voting_config['vote_cutoff'], threshold_pred=hard_voting_config['threshold_pred'])
    dice_score = compute_dice_from_pred_tensor(ensemble_preds, ensemble_gt, threshold_pred=hard_voting_config['threshold_pred'], threshold_gt=hard_voting_config['threshold_gt'])
    print(f"Hard voting mean Dice score: {dice_score.mean()}")

    # soft voting
    ensemble_preds, ensemble_gt = soft_voting(meta_test_loader)
    dice_score = compute_dice_from_pred_tensor(ensemble_preds, ensemble_gt, threshold_pred=soft_voting_config['threshold_pred'], threshold_gt=soft_voting_config['threshold_gt'])
    print(f"Soft voting mean Dice score: {dice_score.mean()}")
    
    # stacking
    # meta_train_loader = DataLoader(meta_train_ds, batch_size=stacking_config['train_config']['batch_size'], shuffle=True)
    # model = train_stacking(meta_train_loader, stacking_config['threshold_gt'], stacking_config['train_config'])
    # # to use a trained model, use the following line:
    # # model = Stacker()
    # # model.load_state_dict(torch.load(stacking_config['model_address']))
    # ensemble_preds, ensemble_gt = stacking(meta_test_loader, model)
    # dice_score = compute_dice_from_pred_tensor(ensemble_preds.squeeze(1), ensemble_gt, threshold_pred=stacking_config['threshold_pred'], threshold_gt=stacking_config['threshold_gt'])
    # print(f"Stacking mean Dice score: {dice_score.mean()}")