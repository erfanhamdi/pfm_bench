import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch
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
    """Run one training epoch."""
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
        # Get a batch of data
        x_batch, y_batch = next(iter(data_loader))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_batch = y_batch > threshold
        # Generate predictions
        y_pred = model(x_batch)
        y_pred = torch.sigmoid(y_pred)
        
        # Log images
        images = []
        for i in range(min(num_samples, len(x_batch))):
            # Create a matplotlib figure for side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Input
            axes[0, 0].imshow(x_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[0, 0].set_title('Input')
            axes[0, 0].axis('off')
            
            # Ground Truth
            axes[0, 1].imshow(y_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            # Prediction
            axes[0, 2].imshow(y_pred[i, 0].cpu().numpy(), cmap='gray')
            axes[0, 2].set_title('Prediction')
            axes[0, 2].axis('off')

            
            plt.tight_layout()
            images.append(wandb.Image(fig, caption=f"{prefix} Sample {i+1}"))
            plt.close(fig)
        
        # Log to wandb
        wandb.log({f"{prefix}_predictions": images})

def training(flnm, epochs, batch_size, learning_rate, decay_rate, decay_step, in_channels, channels, threshold, model_name, 
             alpha=0.1, beta=0.9, seed = 0,run_id=None, wandb_project=None):
    """Run the training process and evaluate on test set."""
    # Setup paths
    base_path = "src/models/unet/results"
    model_dir = f"{base_path}/best_models/{wandb_project}"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if run_id:
        # Resume previous run
        wandb.init(id=run_id, resume="must")
    else:
        # Start a new run
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
                "dataset": flnm,
                "seed": seed
            }
        )
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Create datasets and loaders
    ds = -1
    c = in_channels
    
    train_dataset = UNetDataset(datadir=flnm, split='train', num_c=c, ds_size=ds)
    val_dataset = UNetDataset(datadir=flnm, split='val', num_c=c, apply_transforms=False, ds_size=ds)
    test_dataset = UNetDataset(datadir=flnm, split='test', num_c=c, apply_transforms=False, ds_size=ds)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = UNet(in_channels, channels).double().to(device)
    
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
    best_epoch = -1
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
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
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            best_model_path = f"{model_dir}/{model_name}_best_{epoch}_val_loss_{val_loss:.6f}.pt"
            torch.save(model.state_dict(), best_model_path)
            # Log best model path
            wandb.save(best_model_path)
            wandb.log({"best_val_loss": val_loss, "best_epoch": epoch})
            print(f"Saved new best model with validation loss: {val_loss:.6f}")
        
        # Save intermediate models and log predictions
        if epoch % 30 == 0:
            torch.save(model.state_dict(), f"{model_dir}/aug_{model_name}_intermediate_{epoch}.pt")
            log_prediction_samples(model, val_loader, device, threshold, num_samples=4, prefix="val")
    
    # Save final model
    final_model_path = f"{model_dir}/{model_name}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)
    print(f"Training completed. Final validation loss: {val_losses[-1]:.6f}")
    # Finish wandb run
    wandb.finish()
    
    return train_losses, val_losses

if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
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
    args = parser.parse_args()
    # Train multiple models with different random seeds
    i = args.seed
    flnm = args.data_dir
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
    
    # Train model
    train_losses, val_losses, test_loss, test_dice = training(
        flnm = flnm,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        decay_step=decay_step,
        in_channels=in_channels,
        channels=32,
        threshold=threshold,
        model_name=f'unet-{i}-tension-{args.res}x{args.res}',
        alpha=alpha,  # Weight for Dice loss
        beta=beta,    # Weight for Focal loss
        seed = i,
        wandb_project=f"test-gh-unet-tension-{args.res}x{args.res}-{threshold}"
    )
    
    # print(f"Model {i} test results: Loss={test_loss:.6f}, Dice={test_dice:.6f}", flush=True)