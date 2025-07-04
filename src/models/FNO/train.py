import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import os
import wandb
import matplotlib.pyplot as plt
import glob
from utils import FNODataset
from fno import FNO2d
import argparse

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to: {seed}")

def get_model_path(data_dir: str, seed: int, wandb_project: str) -> str:
    """Generate model path based on dataset name and seed."""
    decomp_name = data_dir.split('/')[-1]
    case_name = data_dir.split('/')[-2]
    model_dir = f"src/models/FNO/results/{wandb_project}/{case_name}_{decomp_name}"
    
    return f"{model_dir}/FNO_{case_name}_{decomp_name}_{seed}.pt"

def log_prediction_samples(true_data, pred_data, epoch: int, 
                         initial_step: int, t_train: int) -> Dict[str, Any]:
   
    images = {}
    num_channels = true_data.shape[-1]
    for ch_idx in range(0, num_channels):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Channel {ch_idx} Comparison (Epoch {epoch})')
        
        sample_idx = 0
        
        # 2 time steps to visualize
        time_steps = [
            initial_step,  
            t_train - 1, 
        ]
        for i, t_idx in enumerate(time_steps):
            # Ground truth
            im1 = axes[i, 0].imshow(true_data[sample_idx,..., t_idx, ch_idx], cmap='coolwarm')
            axes[i, 0].set_title(f'Ground Truth t={t_idx}')
            fig.colorbar(im1, ax=axes[i, 0])
            
            # Prediction
            im2 = axes[i, 1].imshow(pred_data[sample_idx, ..., t_idx, ch_idx], cmap='coolwarm')
            axes[i, 1].set_title(f'Prediction t={t_idx}')
            fig.colorbar(im2, ax=axes[i, 1])
        
            plt.tight_layout()
            images[f'channel_{ch_idx}'] = wandb.Image(fig, caption=f'Channel {ch_idx} (Epoch {epoch})')
            plt.close(fig)
    wandb.log(images, step = epoch)

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
               loss_fn: nn.Module, initial_step: int, t_train: int, training_type: str, device: torch.device) -> Tuple[float, float]:
    """
    Run one training epoch.
    """
    model.train()
    train_l2_step = 0
    train_l2_full = 0
    
    for xx, yy, grid in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)
        if training_type == 'autoregressive':
            loss, batch_l2_full = train_autoregressive(model, xx, yy, grid, initial_step, t_train, loss_fn)
        else:
            raise ValueError(f"Unknown training type: {training_type}")
            
        train_l2_step += loss.item()
        train_l2_full += batch_l2_full
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return train_l2_step, train_l2_full

def train_autoregressive(model: nn.Module, xx: torch.Tensor, yy: torch.Tensor, grid: torch.Tensor, 
                        initial_step: int, t_train: int, loss_fn: nn.Module) -> Tuple[torch.Tensor, float]:
    """Perform autoregressive training."""
    loss = 0
    pred = yy[..., :initial_step, :]
    
    # Reshape input for the model
    inp_shape = list(xx.shape)[:-2]
    inp_shape.append(-1)
    
    for t in range(initial_step, t_train):
        inp = xx.reshape(inp_shape)
        y = yy[..., t:t+1, :]
        
        # Model prediction
        im = model(inp, grid)
        
        # Calculate loss
        _batch = im.size(0)
        loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
        
        # Update predictions and input for next step
        pred = torch.cat((pred, im), -2)
        xx = torch.cat((xx[..., 1:, :], im), dim=-2)
    
    # Calculate full sequence loss
    _batch = yy.size(0)
    _yy = yy[..., :t_train, :]
    l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
    
    return loss, l2_full

def validate(model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module, 
            initial_step: int, t_train: int, epoch: int, device: torch.device) -> float:
    """
    Validate the model and return validation loss.
    
    Args:
        model: The FNO model
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        initial_step: Initial time step
        t_train: Maximum time step to validate
        training_type: 'autoregressive' or 'single'
        
    Returns:
        Validation loss (val_l2_full)
    """
    model.eval()
    val_l2_full = 0
    
    with torch.no_grad():
        for xx, yy, grid in val_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            
            pred = yy[..., :initial_step, :]
            inp_shape = list(xx.shape)[:-2]
            inp_shape.append(-1)
            
            for t in range(initial_step, min(t_train, yy.shape[-2])):
                inp = xx.reshape(inp_shape)
                y = yy[..., t:t+1, :]
                im = model(inp, grid)
                
                pred = torch.cat((pred, im), -2)
                xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            
            _batch = yy.size(0)
            _pred = pred[..., initial_step:t_train, :]
            _yy = yy[..., initial_step:t_train, :]
            val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
        true_data = yy[..., :t_train, :].cpu().numpy()
        pred_data = pred[..., :t_train, :].cpu().numpy()
        log_prediction_samples(true_data, pred_data, epoch, initial_step, t_train)
                
    return val_l2_full

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, model_path: str) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

def load_checkpoint_if_continuing(continue_training: bool, data_dir: str, seed: int, 
                                 wandb_project: str, model: nn.Module, 
                                 optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, float]:
    """
    Load checkpoint if continuing training from a previous run.
    
    Args:
        continue_training: Whether to continue from checkpoint
        data_dir: Dataset directory path
        seed: Random seed used for model naming
        wandb_project: WandB project name for checkpoint organization
        model: The model to load state into
        optimizer: The optimizer to load state into
        device: Device to load tensors onto
        
    Returns:
        Tuple of (start_epoch, min_val_loss)
    """
    start_epoch = 0
    min_val_loss = float('inf')
    
    if not continue_training:
        return start_epoch, min_val_loss
        
    model_path = get_model_path(data_dir, seed, wandb_project)
    print(f"Model path: {model_path}")
    print('Restoring model from checkpoint...')
    
    model_lists = glob.glob(f"{model_path}/*.pt")
    
    if model_lists:
        model_epochs = []
        for model_file in model_lists:
            try:
                epoch_str = model_file.split('_')[-1].split('.')[0]
                epoch_num = int(epoch_str)
                model_epochs.append((epoch_num, model_file))
            except (ValueError, IndexError):
                print(f"Skipping file with invalid format: {model_file}")
        model_epochs.sort(reverse=True)
        
        if model_epochs:
            latest_epoch, latest_model = model_epochs[0]
            print(f"Found latest checkpoint at epoch {latest_epoch}: {latest_model}")
            checkpoint_path = latest_model
        else:
            print("No valid checkpoints found. Starting from scratch.")
            return start_epoch, min_val_loss
    else:
        print("No checkpoints found. Starting from scratch.")
        return start_epoch, min_val_loss
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        start_epoch = checkpoint['epoch'] + 1
        min_val_loss = checkpoint['loss']
        print(f'Starting from epoch {start_epoch} with validation loss {min_val_loss}')
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")
    
    return start_epoch, min_val_loss

def run_training(continue_training: bool,
                 modes: int,
                 width: int,
                 initial_step: int,
                 t_train: int,
                 num_channels: int,
                 epochs: int,
                 learning_rate: float,
                 scheduler_step: int,
                 scheduler_gamma: float,
                 model_update: int,
                 data_dir: str,
                 training_type: str = 'autoregressive',
                 seed: int = 1234,
                 batch_size: int = 10,
                 num_workers: int = 4,
                 use_wandb: bool = True,
                 wandb_project: str = "FNO-Training",
                 model_name: str = None,
                 res: int = 128,
                 **kwargs) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    decomp_name = data_dir.split('/')[-1]
    case_name = data_dir.split('/')[-2]
    model_dir = f"src/models/FNO/results/{wandb_project}/{case_name}_{decomp_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/FNO_{case_name}_{decomp_name}.pt"
    
    if model_name is None:
        model_name = f"FNO_{case_name}_{decomp_name}_{seed}"
        
    wandb.init(
        project=wandb_project,
        name=model_name,
        config={
            "modes": modes,
            "width": width,
            "initial_step": initial_step,
            "t_train": t_train,
            "num_channels": num_channels,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "scheduler_step": scheduler_step,
            "scheduler_gamma": scheduler_gamma,
            "training_type": training_type,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
    )

    # Create datasets and loaders
    train_dataset = FNODataset(data_dir, split='train', initial_step=initial_step, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, ds_size=-1)
    val_dataset = FNODataset(data_dir, split='val', initial_step=initial_step, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, ds_size=-1)
    test_dataset = FNODataset(data_dir, split='test', initial_step=initial_step, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, ds_size=-1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=5, num_workers=num_workers, shuffle=False)

    
    # Initialize model
    _, sample_data, _ = next(iter(train_loader))
    dimensions = len(sample_data.shape) - 3
    print('Spatial Dimension:', dimensions)
    
    # Adjust t_train if needed
    if t_train > sample_data.shape[-2]:
        t_train = sample_data.shape[-2]
        print(f"Adjusted t_train to {t_train}")
    
    # Create model
    model = FNO2d(
        num_channels=num_channels,
        width=width,
        modes1=modes,
        modes2=modes,
        initial_step=initial_step
    ).to(device)
    
    wandb.watch(model, log="all", log_freq=10)
    
    # Initialize optimizer and scheduler
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    if continue_training:
        # Load checkpoint if continuing training
        start_epoch, min_val_loss = load_checkpoint_if_continuing(
            continue_training, data_dir, seed, wandb_project, model, optimizer, device
        )
    else:
        start_epoch = 0
    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}/{epochs}")
        # Train for one epoch
        train_l2_step, train_l2_full = train_epoch(
            model, train_loader, optimizer, loss_fn, initial_step, t_train, training_type, device
            )
        
        # Update learning rate
        scheduler.step()
        
        # Validate periodically
        if epoch % model_update == 0:
            val_l2_full = validate(model, val_loader, loss_fn, initial_step, t_train,epoch, device)
            
            # Save model checkpoint
            checkpoint_path = f"{model_dir}/FNO_{case_name}_{decomp_name}_{epoch}.pt"
            save_checkpoint(model, optimizer,epoch, val_l2_full, checkpoint_path)
            
        else:
            val_l2_full = float('nan')

        print(f'Epoch: {epoch}, Train L2: {train_l2_full:.5f}, Val L2: {val_l2_full:.5f}')
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch":epoch,
                "train_loss": train_l2_full,
                "val_loss": val_l2_full if not np.isnan(val_l2_full) else None,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step = epoch)

if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser(description='Train FNO model')
    parser.add_argument('--initial_step', type=int, default=10, help='Initial time step')
    parser.add_argument('--training_type', type=str, default='autoregressive', help='Training type')
    parser.add_argument('--t_train', type=int, default=101, help='Maximum time step to train')
    parser.add_argument('--model_update', type=int, default=10, help='How often to update the model')
    parser.add_argument('--data_dir', type=str, default='data/tension/spect', help='Path to the dataset file')
    parser.add_argument('--epochs', type=int, default=1001, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for data loaders')
    parser.add_argument('--learning_rate', type=float, default=1.e-3, help='Learning rate')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Steps for learning rate scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Gamma for learning rate scheduler')
    parser.add_argument('--res', type=int, default=128, help='Resolution of the dataset')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of channels')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=20, help='Model width')
    parser.add_argument('--use_wandb', action='store_false', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default="test-gh", help='Wandb project name')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from checkpoint')
    args = parser.parse_args()
    # create a mapping of args to kwargs
    kwargs = {
        'initial_step': args.initial_step,
        't_train': args.t_train,
        'model_update': args.model_update,
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'scheduler_step': args.scheduler_step,
        'scheduler_gamma': args.scheduler_gamma,
        'num_channels': args.num_channels,
        'modes': args.modes,
        'width': args.width,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'continue_training': args.continue_training,
        'seed': args.seed,
        'res': args.res,
        'training_type': args.training_type,
    }
    set_seed(args.seed)
    if args.continue_training:
        print('Continuing training from checkpoint...')
    run_training(**kwargs)
    
    # Finish wandb run
    if not args.use_wandb:
        wandb.finish()
    
    print("Training completed.")