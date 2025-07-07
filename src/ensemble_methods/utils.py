import h5py
import numpy as np

class EnsembleDataset:
    def __init__(self, pred_array, gt_array, keys):
        self.pred_array = pred_array
        self.gt_array = gt_array
        self.keys = keys
        self.pred_array_permuted = pred_array.transpose(1, 0, 2, 3)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.keys[idx], self.pred_array_permuted[idx], self.gt_array[idx]

# def load_predictions(seed_pred_list):
#     pred_list = []
#     keys_list = []
#     gt_list = []
#     for pred_dict_dir in seed_pred_list:
#         with open(pred_dict_dir, 'rb') as f:
#             pred_dict = pickle.load(f)
        
#         keys = list(pred_dict['pred'].keys())
#         if pred_dict['pred'][keys[0]].shape[3] > 1:
#             pred_values_stacked = torch.stack(list(pred_dict['pred'].values()))[..., -1, 0:1].squeeze(1).unsqueeze(-2)
#             gt_values_stacked = torch.stack(list(pred_dict['gt'].values()))[..., -1, 0:1].squeeze(1).unsqueeze(-2)
#         else:
#             pred_values_stacked = torch.stack(list(pred_dict['pred'].values())).unsqueeze(1)
#             gt_values_stacked = torch.stack(list(pred_dict['gt'].values())).unsqueeze(1)
#         keys_list.append(keys)
#         pred_list.append(pred_values_stacked)
#         gt_list.append(gt_values_stacked)
#         # delete pred_dict
#         del pred_dict
#     pred_array = torch.stack(pred_list)
#     gt_array = torch.stack(gt_list)[0]
#     final_keys = keys_list[0]
    
#     return final_keys, pred_array, gt_array

def load_predictions(seed_pred_list):
    pred_list = []
    for pred_dict_dir in seed_pred_list:
        with h5py.File(pred_dict_dir, 'r') as f:
            pred_list.append(f['pred'][:])        
            gt_array = f['gt'][:]
            key_list = f['keys'][:]
    pred_array = np.stack(pred_list)[..., 0]
    gt_array = gt_array[..., 0]
    return key_list, pred_array, gt_array