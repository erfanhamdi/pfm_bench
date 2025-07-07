import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

def soft_dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1,2,3))
    union = (p.pow(2) + target.pow(2)).sum(dim=(1,2,3))
    dice = 1 - (2*inter + eps) / (union + eps)
    return dice.mean()

class Stacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv2d(5, 5, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(5, 5, 3, padding=1, groups=5, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(5, 1, kernel_size=1, bias=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.w1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.fuse(x)
        return x

    def _initialize_weights(self,):
        with torch.no_grad():
            init.xavier_uniform_(self.w1.weight)
            self.w1.bias.data.fill_(0.2)
            self.dw.weight.zero_()
            self.dw.weight[..., 1, 1] = 1.0
            init.xavier_uniform_(self.fuse.weight)
            self.fuse.bias.zero_()

def train_stacking(train_loader, threshold_gt, train_config):
    torch.manual_seed(42)
    np.random.seed(42)
    model = Stacker()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    loss_fn = soft_dice_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=train_config['patience']) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(train_config['epochs']):
        train_loss = 0.0
        for _, xb, yb in train_loader:
            yb = (yb > threshold_gt).float()
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if yb.ndim == 3:
                yb = yb.unsqueeze(1)
            xb = xb.float()
            pred_logits = model(xb)
            loss = loss_fn(pred_logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)
    torch.save(model.state_dict(), f'{train_config['out_dir']}/stacker_model.pth')
    return model