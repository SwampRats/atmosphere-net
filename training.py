import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from atmosphere_net import AtmosphereNet
import json

# Data generation function (replace this with your actual atmosphere function)
def read_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        inputs = np.array([item['input'] for item in data])
        outputs = np.array([item['output'] for item in data])
    return inputs.astype(np.float32), outputs.astype(np.float32)


def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate or load your data
    print("Preparing training data...")
    X, y = read_training_data("atmosphere_dataset.json")

    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    model = AtmosphereNet().to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler (optional but recommended)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.6f} "
                  f"Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_atmosphere_model.pth')
            print(f"  New best model saved! Val Loss: {best_val_loss:.6f}")
    
    print("-" * 60)
    print("Training completed!")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, train_losses, val_losses

def test_model():
    """Test the trained model on some sample inputs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the best model
    model = AtmosphereNet().to(device)
    model.load_state_dict(torch.load('best_atmosphere_model.pth'))
    model.eval()
    
    # Test on some sample inputs
    test_inputs = np.array([
        [0.0, 1.0, 0.0],  # Looking up
        [1.0, 0.0, 0.0], # Looking horizon
        [0.0, 0.0, 1.0],  # Looking forward
    ])
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_inputs).to(device)
        predictions = model(test_tensor).cpu().numpy()
    
    print("Sample predictions:")
    for i, (input_data, color) in enumerate(zip(test_inputs, predictions)):
        print(f"Input {i+1}: ray={input_data[:3]}, sun={input_data[3:]}")
        print(f"  Predicted RGB: ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
        print()

# Example usage
if __name__ == "__main__":
    # Train the model
    model, train_losses, val_losses = train_model()

    
    # Test the trained model
    test_model()
    
    print("Training complete! Model saved as 'best_atmosphere_model.pth'")