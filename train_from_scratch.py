"""
Training Script for Semiconductor Image Restoration
This script trains a super-resolution model from scratch on the training dataset.
USAGE: python train_from_scratch.py --data_dir <path_to_train_folder>
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from model import SuperResolutionNet

class SemiconDataset(Dataset):
    """Load training pairs"""
    def __init__(self, gt_dir, noisy_dir):
        self.gt_dir = Path(gt_dir)
        self.noisy_dir = Path(noisy_dir)
        self.files = sorted(self.gt_dir.glob('*.npy'))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx].name
        gt = np.load(self.gt_dir / filename).astype(np.float32)
        noisy = np.load(self.noisy_dir / filename).astype(np.float32)
        gt = torch.from_numpy(gt).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        return noisy, gt

def train(data_dir, num_epochs=10, batch_size=16, learning_rate=0.001):
    """Train super-resolution model"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    gt_path = Path(data_dir) / 'GT'
    noisy_path = Path(data_dir) / 'NoisyLR'
    
    if not gt_path.exists() or not noisy_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_dir}")
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = SemiconDataset(gt_path, noisy_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"✓ Loaded {len(dataset)} training pairs")
    
    # Create model
    print("\n[2/4] Creating model...")
    model = SuperResolutionNet().to(device)
    print("✓ Model created")
    
    # Setup training
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    best_loss = float('inf')
    
    # Training loop
    print("\n[3/4] Starting training...")
    print("="*60)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (noisy, gt) in enumerate(dataloader):
            noisy = noisy.to(device)
            gt = gt.to(device)
            
            output = model(noisy)
            loss = criterion(output, gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(dataloader)
        elapsed_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {elapsed_time:.1f}s")
        print(f"Average Loss: {avg_loss:.6f}")
        print("-" * 60)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Saved best model (loss: {best_loss:.6f})\n")
    
    print("="*60)
    print("[4/4] Training complete!")
    print(f"\n✓ Best model saved as 'best_model.pth'")
    print(f"✓ Best loss: {best_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Super-Resolution Model'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to train folder containing GT and NoisyLR subfolders'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    train(args.data_dir, args.num_epochs, args.batch_size, args.learning_rate)