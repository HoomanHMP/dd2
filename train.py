import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from data import get_datasets
from model import SudokuCNN


class MultiTaskLoss(nn.Module):
    """Combined loss for all three tasks"""
    def __init__(self, weight_missing=1.0, weight_sorted=1.0, weight_sum=1.0):
        super(MultiTaskLoss, self).__init__()
        self.weight_missing = weight_missing
        self.weight_sorted = weight_sorted
        self.weight_sum = weight_sum
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn. MSELoss()
    
    def forward(self, predictions, targets):
        # Missing digit loss
        loss_missing = self.ce_loss(predictions['missing_digit'], targets['missing_digit'])
        
        # Sortedness loss (6 independent classifications)
        loss_sorted = 0
        for i in range(6):
            loss_sorted += self.ce_loss(predictions['sorted_labels'][:, i, :], 
                                       targets['sorted_labels'][:, i])
        loss_sorted /= 6
        
        # Sum regression loss
        loss_sum = self.mse_loss(predictions['sum_labels'], targets['sum_labels'])
        
        # Combined loss
        total_loss = (self.weight_missing * loss_missing + 
                     self. weight_sorted * loss_sorted + 
                     self.weight_sum * loss_sum)
        
        return total_loss, loss_missing, loss_sorted, loss_sum


def calculate_metrics(predictions, targets):
    """Calculate accuracy and MAE for all tasks"""
    with torch.no_grad():
        # Missing digit accuracy
        missing_pred = torch.argmax(predictions['missing_digit'], dim=1)
        missing_acc = (missing_pred == targets['missing_digit']).float().mean().item()
        
        # Sortedness accuracy (average across 6 positions)
        sorted_pred = torch.argmax(predictions['sorted_labels'], dim=2)  # (batch, 6)
        sorted_acc = (sorted_pred == targets['sorted_labels']).float().mean().item()
        
        # Sum MAE
        sum_mae = torch.abs(predictions['sum_labels'] - targets['sum_labels']).mean().item()
    
    return missing_acc, sorted_acc, sum_mae


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_missing_acc = 0
    total_sorted_acc = 0
    total_sum_mae = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        images = batch['image'].to(device)
        targets = {
            'missing_digit': batch['missing_digit'].to(device),
            'sorted_labels': batch['sorted_labels'].to(device),
            'sum_labels':  batch['sum_labels'].to(device)
        }
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Calculate loss
        loss, loss_missing, loss_sorted, loss_sum = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        missing_acc, sorted_acc, sum_mae = calculate_metrics(predictions, targets)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_missing_acc += missing_acc
        total_sorted_acc += sorted_acc
        total_sum_mae += sum_mae
        num_batches += 1
        
        # Update progress bar
        progress_bar. set_postfix({
            'loss': f'{loss.item():.4f}',
            'missing_acc': f'{missing_acc:.4f}',
            'sorted_acc': f'{sorted_acc:. 4f}',
            'sum_mae': f'{sum_mae:.4f}'
        })
    
    return (total_loss / num_batches, 
            total_missing_acc / num_batches,
            total_sorted_acc / num_batches,
            total_sum_mae / num_batches)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_missing_acc = 0
    total_sorted_acc = 0
    total_sum_mae = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move data to device
            images = batch['image'].to(device)
            targets = {
                'missing_digit': batch['missing_digit']. to(device),
                'sorted_labels': batch['sorted_labels'].to(device),
                'sum_labels': batch['sum_labels'].to(device)
            }
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss, _, _, _ = criterion(predictions, targets)
            
            # Calculate metrics
            missing_acc, sorted_acc, sum_mae = calculate_metrics(predictions, targets)
            
            # Accumulate metrics
            total_loss += loss. item()
            total_missing_acc += missing_acc
            total_sorted_acc += sorted_acc
            total_sum_mae += sum_mae
            num_batches += 1
    
    return (total_loss / num_batches,
            total_missing_acc / num_batches,
            total_sorted_acc / num_batches,
            total_sum_mae / num_batches)


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history with train, validation, and test metrics"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    if 'test_loss' in history:
        ax.plot(len(epochs), history['test_loss'], 'go', markersize=10, label='Test Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Missing Digit Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_missing_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs, history['val_missing_acc'], 'r-', label='Val Accuracy', linewidth=2)
    if 'test_missing_acc' in history:
        ax.plot(len(epochs), history['test_missing_acc'], 'go', markersize=10, label='Test Accuracy')
    ax.axhline(y=0.95, color='g', linestyle='--', label='Target (95%)', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Missing Digit Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 3: Sortedness Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['train_sorted_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs, history['val_sorted_acc'], 'r-', label='Val Accuracy', linewidth=2)
    if 'test_sorted_acc' in history:
        ax. plot(len(epochs), history['test_sorted_acc'], 'go', markersize=10, label='Test Accuracy')
    ax.axhline(y=0.93, color='g', linestyle='--', label='Target (93%)', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Sortedness Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 4: Sum MAE
    ax = axes[1, 1]
    ax.plot(epochs, history['train_sum_mae'], 'b-', label='Train MAE', linewidth=2)
    ax.plot(epochs, history['val_sum_mae'], 'r-', label='Val MAE', linewidth=2)
    if 'test_sum_mae' in history:
        ax.plot(len(epochs), history['test_sum_mae'], 'go', markersize=10, label='Test MAE')
    ax.axhline(y=0.06, color='r', linestyle='--', label='Target (0.06)', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Row/Column Sum MAE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ROOT_DIR = './data'
    
    print(f"Using device: {DEVICE}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(ROOT_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = SudokuCNN().to(DEVICE)
    print(f"Total parameters: {model.count_parameters()}")
    
    # Loss and optimizer
    criterion = MultiTaskLoss(weight_missing=1.0, weight_sorted=1.0, weight_sum=10.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler. ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_missing_acc': [], 'val_missing_acc': [],
        'train_sorted_acc': [], 'val_sorted_acc': [],
        'train_sum_mae': [], 'val_sum_mae':  []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_missing_acc, train_sorted_acc, train_sum_mae = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_missing_acc, val_sorted_acc, val_sum_mae = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss']. append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_missing_acc'].append(train_missing_acc)
        history['val_missing_acc']. append(val_missing_acc)
        history['train_sorted_acc'].append(train_sorted_acc)
        history['val_sorted_acc'].append(val_sorted_acc)
        history['train_sum_mae'].append(train_sum_mae)
        history['val_sum_mae']. append(val_sum_mae)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Missing Acc: {train_missing_acc:.4f} | Val Missing Acc: {val_missing_acc:.4f}")
        print(f"Train Sorted Acc: {train_sorted_acc:.4f} | Val Sorted Acc: {val_sorted_acc:.4f}")
        print(f"Train Sum MAE: {train_sum_mae:.4f} | Val Sum MAE: {val_sum_mae:.4f}")
        
        # Save best model
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ Best model saved (val_loss:  {val_loss:.4f})")
    
    # Load best model for testing
    print("\n" + "=" * 50)
    print("Training completed!  Loading best model for testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_missing_acc, test_sorted_acc, test_sum_mae = validate(
        model, test_loader, criterion, DEVICE
    )
    
    # Add test results to history
    history['test_loss'] = test_loss
    history['test_missing_acc'] = test_missing_acc
    history['test_sorted_acc'] = test_sorted_acc
    history['test_sum_mae'] = test_sum_mae
    
    # Print final results
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Missing Digit Accuracy: {test_missing_acc:.4f} (Target: ≥0.95) {'✓' if test_missing_acc >= 0.95 else '✗'}")
    print(f"Sortedness Accuracy: {test_sorted_acc:. 4f} (Target: ≥0.93) {'✓' if test_sorted_acc >= 0.93 else '✗'}")
    print(f"Sum MAE: {test_sum_mae:.4f} (Target: ≤0.06) {'✓' if test_sum_mae <= 0.06 else '✗'}")
    print("=" * 50)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
