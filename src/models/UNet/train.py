import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
import wandb
from unet import UNet, CombinedLoss
from utils import UNetDataset
import argparse

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to: {seed}", flush=True)

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    Run one training epoch.
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    for x_batch, y_batch in train_loader:
        n_batches += 1
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        outputs = model(x_batch)
        loss = loss_fn(outputs, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * len(x_batch)
        
        if n_batches % 10 == 0:
            print(f"Batch {n_batches} | Loss: {batch_loss:.6f}", flush=True)
            
    return total_loss

def validate(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item() * len(x_batch)
            
    return total_loss

def log_prediction_samples(model, data_loader, device, threshold, num_samples=4, prefix="val"):
    """Log prediction samples to wandb."""
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(data_loader))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_batch = y_batch > threshold
        y_pred = model(x_batch)
        y_pred = torch.sigmoid(y_pred)
        
        images = []
        for i in range(min(num_samples, len(x_batch))):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Input
            axes[0].imshow(x_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')
            
            # Ground Truth
            axes[1].imshow(y_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(y_pred[i, 0].cpu().numpy(), cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            images.append(wandb.Image(fig, caption=f"{prefix} Sample {i+1}"))
            plt.close(fig)
        
        # Log to wandb
        wandb.log({f"{prefix}_predictions": images})

def run_training(data_dir: str, 
                 epochs: int, 
                 batch_size: int, 
                 learning_rate: float, 
                 decay_rate: float, 
                 decay_step: int, 
                 in_channels: int, 
                 channels: int, 
                 threshold: float, 
                 alpha: float = 0.1, 
                 beta: float = 0.9, 
                 seed: int = 0,
                 wandb_project: str = "UNet-Training",
                 model_name: str = None):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    decomp_name = data_dir.split('/')[-1]
    case_name = data_dir.split('/')[-2]
    model_dir = f"src/models/UNet/results/{wandb_project}/{case_name}_{decomp_name}/seed_{seed}"
    # TODO: have the same model saving structure for the FNO
    model_path = f"{model_dir}/models"
    os.makedirs(model_path, exist_ok=True)
    
    wandb.init(
        project=wandb_project,
        name=model_name,
        config={
            "learning_rate": learning_rate,
            "decay_rate": decay_rate,
            "decay_step": decay_step,
            "in_channels": in_channels,
            "channels": channels, "threshold":threshold,
            "alpha": alpha,
            "beta": beta,
            "epochs": epochs,
            "architecture": "UNet",
            "loss": f"CombinedLoss(alpha={alpha}, beta={beta})",
            "dataset": data_dir,
            "seed": seed
        }
    )
    
    # Create datasets and loaders
    train_dataset = UNetDataset(datadir=data_dir, split='train', num_c=in_channels, ds_size=-1)
    val_dataset = UNetDataset(datadir=data_dir, split='val', num_c=in_channels, apply_transforms=False, ds_size=-1)
    test_dataset = UNetDataset(datadir=data_dir, split='test', num_c=in_channels, apply_transforms=False, ds_size=-1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = UNet(in_channels, channels).to(device)
    
    # Log model architecture to wandb
    wandb.watch(model, log="all", log_freq=10)
    
    # Setup loss, optimizer and scheduler
    loss_fn = CombinedLoss(alpha=alpha, beta=beta, threshold=threshold).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    
    # Training loop
    min_val_loss = float('inf')
    train_losses = []
    val_losses = []
    best_model_path = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device
            )
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        })
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.8f}", flush=True)
        
        # Save best model
        # TODO: saving the best model should be the same as FNO
        # Save intermediate models and log predictions
        if epoch % 30 == 0:
            torch.save(model.state_dict(), f"{model_path}/UNet_epoch_{epoch}.pt")
            log_prediction_samples(model, val_loader, device, threshold, num_samples=4, prefix="val")
    
    # Save final model
    final_model_path = f"{model_path}/UNet_final.pt"
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    print(f"Training completed. Final validation loss: {val_losses[-1]:.6f}")
    # Finish wandb run
    wandb.finish()
    
    return train_losses, val_losses

if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--decay_rate", type=float, default=1)
    parser.add_argument("--decay_step", type=int, default=50)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--data_dir", type=str, default="data/tension/spect")
    parser.add_argument("--wandb_project", type=str, default="test-UNet-Training")
    args = parser.parse_args()
    # Train multiple models with different random seeds
    i = args.seed
    data_dir = args.data_dir
    epochs = args.epochs
    threshold = args.threshold
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    decay_step = args.decay_step
    in_channels = args.in_channels
    alpha = args.alpha
    beta = args.beta
    print(f'\n=== Training model {i} ===', flush=True)
    set_seed(i)
    case_name = data_dir.split('/')[-2]
    decomp_name = data_dir.split('/')[-1]
    model_name = f"UNet_{case_name}_{decomp_name}_{i}"
    # Train model
    train_losses, val_losses, test_loss, test_dice = run_training(
        data_dir = data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        decay_step=decay_step,
        in_channels=in_channels,
        channels=32,
        threshold=threshold,
        model_name=model_name,
        alpha=alpha,  # Weight for Dice loss
        beta=beta,    # Weight for Focal loss
        seed = i,
        wandb_project=args.wandb_project
    )
    