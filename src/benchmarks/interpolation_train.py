import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from nn.UNet import UNet
from nn.Datasets import LazyInterpolationDataset
from piqa import SSIM
from scipy.stats import pearsonr

class SSIMLoss(SSIM):
    def __init__(self, window_size = 11, sigma = 1.5, n_channels = 1, reduction = 'mean', **kwargs):
        super().__init__(window_size, sigma, n_channels, reduction, **kwargs)

    def forward(self, x, y):
        return 1. - super().forward(x, y)

class PCCLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.view(pred.shape[0], -1)  # Flatten
        target = target.view(target.shape[0], -1)
        mean_pred = pred.mean(dim=1, keepdim=True)
        mean_target = target.mean(dim=1, keepdim=True)
        
        num = ((pred - mean_pred) * (target - mean_target)).sum(dim=1)
        denom = torch.sqrt(((pred - mean_pred) ** 2).sum(dim=1) * ((target - mean_target) ** 2).sum(dim=1))
        
        pcc = num / (denom + 1e-8)  # Avoid division by zero
        return 1 - pcc.mean()  # Convert similarity to loss

def normalize_tensor(tensor):
    if tensor.dim() == 3:  # Shape (batch, height, width), missing channels
        tensor = tensor.unsqueeze(1)  # Add channel dimension (batch, 1, height, width)

    min_val = tensor.amin(dim=(2, 3), keepdim=True)  # Global min across spatial dimensions
    max_val = tensor.amax(dim=(2, 3), keepdim=True)  # Global max

    return (tensor - min_val) / (max_val - min_val + 1e-8)  # Normalize to [0,1]

def evaluate(model, test_dataloader, ssim_loss, pcc_loss, device):
    model.eval()
    ssim_list = []
    pcc_list = []
    
    with torch.no_grad():
        for inputs, target in test_dataloader:
            inputs, target = inputs.to(device), target.to(device)

            outputs = model(inputs)  # Expected shape: [batch, height, width]
            if outputs.dim() == 3:  # Add channel dimension if missing
                outputs = outputs.unsqueeze(1)
            target = target.unsqueeze(1) if target.dim() == 3 else target
            
            # Normalize tensors before evaluation
            outputs = normalize_tensor(outputs)
            target = normalize_tensor(target)
            
            # Compute SSIM and PCC
            for out, tgt in zip(outputs, target):
                # Ensure tensors are detached and on the correct device
                out_tensor = out.clone().detach().unsqueeze(0).to(device)
                tgt_tensor = tgt.clone().detach().unsqueeze(0).to(device)
                
                ssim_list.append(ssim_loss(out_tensor, tgt_tensor).item())  

                # Move tensors to CPU before converting to numpy for pearsonr
                pcc_list.append(pearsonr(out_tensor.cpu().numpy().flatten(), 
                                        tgt_tensor.cpu().numpy().flatten())[0])
                
    # Compute statistics
    ssim_array = np.array(ssim_list)
    pcc_array = np.array(pcc_list)
    
    print(f"SSIM - Mean: {ssim_array.mean():.4f}, Std: {ssim_array.std():.4f}, "
          f"25th: {np.percentile(ssim_array, 25):.4f}, "
          f"50th (Median): {np.percentile(ssim_array, 50):.4f}, "
          f"75th: {np.percentile(ssim_array, 75):.4f}")
    
    print(f"PCC - Mean: {pcc_array.mean():.4f}, Std: {pcc_array.std():.4f}, "
          f"25th: {np.percentile(pcc_array, 25):.4f}, "
          f"50th (Median): {np.percentile(pcc_array, 50):.4f}, "
          f"75th: {np.percentile(pcc_array, 75):.4f}")



def train(model, train_dataloader, optimizer, ssim_loss, pcc_loss, device, epochs, alpha=0.5):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0

        for inputs, target in train_dataloader:
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            outputs = normalize_tensor(outputs)
            target = normalize_tensor(target)

            loss = alpha * ssim_loss(outputs, target) + (1 - alpha) * pcc_loss(outputs, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_dataloader):.4f}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    epochs = 5
    lr = 1e-4
    alpha = 0.1

    trs = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)), transforms.ToTensor()])

    dataset_path = "/mnt/d/d_merged12/A-TGF/"
    dataset = LazyInterpolationDataset(dataset_path, device, transform=trs)

    # Split into train (80%) and test (20%) sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ssim_loss = SSIMLoss().to(device)
    pcc_loss = PCCLoss().to(device)

    train(model, train_dataloader, optimizer, ssim_loss, pcc_loss, device, epochs, alpha)
    torch.save(model.state_dict(), "unet_interpolation.th")
    evaluate(model, test_dataloader, ssim_loss, pcc_loss, device)

if __name__ == '__main__':
    main()
