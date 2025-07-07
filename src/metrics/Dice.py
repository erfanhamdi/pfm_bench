import pickle
import os
import numpy as np
import h5py
import torch

def load_pred_dict(pred_dict_path):
    with h5py.File(pred_dict_path, 'r') as f:
        pred_dict = {key: f[key][:] for key in f.keys()}
    return pred_dict


def dice(pred, gt):
    intersection = torch.sum(pred * gt, axis=(1, 2))
    return (2. * intersection) / (torch.sum(pred, axis=(1, 2)) + torch.sum(gt, axis=(1, 2)))

def compute_dice_from_pred_dict(pred_dict_path, out_dir, threshold_pred=0.5, threshold_gt=0.5):
    pred_dict = load_pred_dict(pred_dict_path)
    pred = (pred_dict['pred'][..., 0] > threshold_pred).astype(np.float32)
    gt = (pred_dict['gt'][..., 0] > threshold_gt).astype(np.float32)
    dice_scores = dice(pred, gt)
    with open(os.path.join(out_dir, "dice_scores.pkl"), "wb") as f:
        pickle.dump(dice_scores, f)
    return dice_scores

def compute_dice_from_pred_tensor(pred_tensor, gt_tensor, threshold_pred=0.5, threshold_gt=0.5):
    pred = (pred_tensor > threshold_pred).float()
    gt = (gt_tensor > threshold_gt).float()
    return dice(pred, gt)
