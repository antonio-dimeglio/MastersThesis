from nn.Datasets import CollagenDataset
from nn.UNet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics import Dice
from sklearn.metrics import jaccard_score
from tqdm import tqdm


def evaluate(model, test_dataloader, device):
    model.eval()  # Set to evaluation mode
    dice_metric = Dice().to(device)
    iou_metric = jaccard_score

    total_dice = 0.0
    total_iou = 0.0
    num_batches = len(test_dataloader)

    with torch.no_grad():  # No gradient computation during evaluation
        for images, masks in test_dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)  # Shape: (batch_size, H, W)
            outputs = (torch.sigmoid(outputs) > 0.5).float()  # Convert to binary mask
            masks = masks.squeeze(1)

            # Convert masks to integers (0 or 1)
            masks = masks.int()

            # Calculate Dice coefficient
            total_dice += dice_metric(outputs, masks).item()

            # Calculate IoU (Jaccard Index)
            outputs = outputs.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()
            total_iou += iou_metric(outputs, masks)

    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    print(f"Test Results - Dice Coefficient: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

def train(model, train_dataloader, optimizer, criterion, device, epochs):
    model.train()  # Set to training mode
    epoch_losses = []  # To store loss values for plotting

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        batch_progress = tqdm(train_dataloader, desc="Batch Progress", leave=False, position=0)
        
        for images, masks in batch_progress:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)  # Ensure output shape matches mask
            masks = masks.squeeze(1)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_progress.set_postfix(loss=loss.item())

        # Store average loss per epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
        

def main():
    # Hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    epochs = 5
    lr = 1e-4

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = CollagenDataset(
        "/home/quantum/MScThesisCode/dataset/images",
        "/home/quantum/MScThesisCode/dataset/ground",
        device=device,
        transform=transform
    )


    # Split into train (80%) and test (20%) sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(n_channels=1, n_classes=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)


    train(model, train_dataloader, optimizer, criterion, device, epochs)
    evaluate(model, test_dataloader, device)
    torch.save(model.state_dict(), "unet.th")

if __name__ == '__main__':
    main()