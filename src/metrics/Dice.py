import pickle
import os
import torch

def load_pred_dict(pred_dict_path):
    with open(pred_dict_path, 'rb') as f:
        pred_dict = pickle.load(f)
    return pred_dict


def dice(pred, gt):
    intersection = torch.sum(pred * gt)
    return (2. * intersection) / (torch.sum(pred) + torch.sum(gt))

def compute_dice(pred_dict_path, out_dir, threshold_pred=0.5, threshold_gt=0.5):
    pred_dict = load_pred_dict(pred_dict_path)
    dice_scores = {}
    for seed in pred_dict['init'].keys():
        pred = (pred_dict['pred'][seed] > threshold_pred).float()
        gt = (pred_dict['gt'][seed] > threshold_gt ).float()
        dice_scores[seed] = dice(pred, gt)
    with open(os.path.join(out_dir, "dice_scores.pkl"), "wb") as f:
        pickle.dump(dice_scores, f)
    return dice_scores
