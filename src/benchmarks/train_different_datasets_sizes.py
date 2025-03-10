import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from nn.Datasets import CollagenDataset
from nn.UNet import UNet
from tqdm import tqdm
from collagen_train import evaluate

# Hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 5
lr = 1e-4

# Define transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load dataset
dataset = CollagenDataset(
    "/home/quantum/MScThesisCode/dataset/images",
    "/home/quantum/MScThesisCode/dataset/ground",
    device=device,
    transform=transform
)

# Define different training percentages
train_percentages = [0.3, 0.5, 0.6, 0.7, 0.8]
results = {}

total_size = len(dataset)

for train_percent in train_percentages:
    train_size = int(train_percent * total_size)
    print(f"Training with {train_percent * 100:.0f}% ({train_size} samples)")
    
    train_dataset, test_dataset = random_split(dataset, [train_size, total_size - train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        batch_progress = tqdm(train_dataloader, desc="Batch Progress", leave=False, position=0)
        
        for images, masks in batch_progress:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)
            masks = masks.squeeze(1)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_progress.set_postfix(loss=loss.item())
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")
    
    # Evaluate
    print("Evaluating...")
    evaluate(model, test_dataloader, device)
    results[train_size] = model.state_dict()

    # Generate and save prediction plot
    model.eval()
    with torch.no_grad():
        sample_image, sample_mask = test_dataset[np.random.randint(len(test_dataset))]
        sample_image = sample_image.unsqueeze(0).to(device)
        pred_mask = torch.sigmoid(model(sample_image))
        pred_mask = (pred_mask > 0.5).long().cpu().squeeze().numpy()  # Convert to numpy
        sample_image = sample_image.squeeze().cpu().numpy()
        sample_mask = sample_mask.squeeze().cpu().numpy()  # Ensure it's 2D

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(sample_image, cmap='gray')
        axs[0].set_title("Original Image")
        axs[1].imshow(sample_mask, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_mask, cmap='gray')
        axs[2].set_title("Predicted Mask")

        for ax in axs:
            ax.axis("off")

        plt.savefig(f"prediction_{train_percent * 100:.0f}percent.png", bbox_inches='tight')
        plt.close()
    
print("Training complete. Models and plots saved.")
