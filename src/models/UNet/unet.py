import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_c, c):
        super(UNet, self).__init__()
        # Encoder blocks
        self.enc1 = DoubleConv(in_c, c)
        self.enc2 = DoubleConv(c, 2*c)
        self.enc3 = DoubleConv(2*c, 4*c)
        self.enc4 = DoubleConv(4*c, 8*c)
        
        # Decoder blocks
        self.dec3 = DoubleConv(8*c, 4*c)
        self.dec2 = DoubleConv(4*c, 2*c)
        self.dec1 = DoubleConv(2*c, c)
        
        # Upsampling
        self.upconv3 = nn.ConvTranspose2d(8*c, 4*c, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(4*c, 2*c, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(2*c, c, kernel_size=2, stride=2)

        # Output layer
        self.out = nn.Conv2d(c, in_c, kernel_size=3, padding=1)
        
        # Pooling
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.maxpool(x1)
        
        x2 = self.enc2(x)
        x = self.maxpool(x2)
        
        x3 = self.enc3(x)
        x = self.maxpool(x3)
        
        x = self.enc4(x)
        
        # Decoder with skip connections
        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        return self.out(x)

class DiceLoss(nn.Module):
    """Dice Loss for sparse segmentation tasks."""
    def __init__(self, smooth=1e-6, threshold=0.4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        targets = (targets > self.threshold).float()
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice_score  # Dice Loss

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.8, gamma=2, threshold=0.4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        targets = (targets > self.threshold).float()  # Convert boolean to float
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
        focal_loss = self.alpha * (1 - preds) ** self.gamma * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined Dice Loss + Focal Loss for sparse segmentation."""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, gamma=2, threshold=0.4):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth, threshold)
        self.focal_loss = FocalLoss(gamma=gamma, threshold=threshold)
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        dice = self.dice_loss(preds, targets)
        focal = self.focal_loss(preds, targets)
        return self.alpha * dice + self.beta * focal  # Weighted sum of losses
