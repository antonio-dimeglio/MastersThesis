import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import tifffile
import numpy as np

class CollagenDataset(Dataset):
    def __init__(self, image_dir:str, ground_dir:str,  device:str, transform=None):
        super().__init__()

        self.transform = transform 
        self.image_dir = image_dir
        self.ground_dir = ground_dir
        self.device = device 
        
        # Get the list of all image file names (assuming all files in images match those in ground)
        self.image_files = os.listdir(self.image_dir)
    
    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.image_files)
    
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: A dictionary containing 'image' and 'ground', which are the input image 
                  and corresponding ground truth label.
        """
        # Get the filename for the image and the corresponding ground truth
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        ground_path = os.path.join(self.ground_dir, image_name)
     
        # Load image
        image = Image.open(image_path).convert("L")
        
        # Load ground truth
        ground = Image.open(ground_path).convert("L")   # Or use np.load if ground is a .npy file
        
        # Convert image and ground to tensors if needed
        if self.transform:
            image = self.transform(image)  # Apply the transformation to the image
            ground = self.transform(ground)
            ground = (ground > 0.5).float()  

        image = image.to(self.device)
        ground = ground.to(self.device)
        
        return image, ground

class LazyInterpolationDataset(Dataset):
    def __init__(self, dataset_path: str, device: str, transform=None):
        super().__init__()
        self.device = device
        self.transform = transform
        self.file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        self.data_indices = []  # Store (file_path, slice_index) instead of loading images
        
        for file_path in self.file_paths:
            stack = tifffile.memmap(file_path)  # Memory map instead of loading
            for i in range(1, stack.shape[0] - 1):
                self.data_indices.append((file_path, i))  # Store file path and index
        
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        file_path, i = self.data_indices[idx]
        stack = tifffile.imread(file_path)  # Load only when needed
        
        img_prev = stack[i - 1].astype(np.float32)
        img_next = stack[i + 1].astype(np.float32)
        img_target = stack[i].astype(np.float32)
        img_interp = np.linspace(img_prev, img_next, 3)[1].astype(np.float32)

        # Stack inputs into a single tensor of shape (3, H, W)
        input_tensor = np.stack([img_prev, img_next, img_interp], axis=0)  # Shape (3, H, W)
        target_tensor = img_target[np.newaxis, ...]  # Shape (1, H, W)

        # Convert to torch tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float16)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float16)

        # Apply optional transformation
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor.to(self.device), target_tensor.to(self.device)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Initialize dataset
    dataset_path = "/mnt/d/d_merged12/A-TGF/"
    device = "cuda"  # or "cuda" if using GPU
    dataset = InterpolationDataset(dataset_path, device)

    # Get a single sample (e.g., first one)
    (inputs, target) = dataset[0]
    img_prev, img_next, img_interp = inputs  # Unpack input images

    # Convert tensors to numpy for visualization
    img_prev_np = img_prev.squeeze().cpu().numpy()
    img_next_np = img_next.squeeze().cpu().numpy()
    img_interp_np = img_interp.squeeze().cpu().numpy()
    img_target_np = target.squeeze().cpu().numpy()

    # Plot images
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_prev_np, cmap="gray")
    axes[0].set_title("Previous Slice")
    axes[1].imshow(img_next_np, cmap="gray")
    axes[1].set_title("Next Slice")
    axes[2].imshow(img_interp_np, cmap="gray")
    axes[2].set_title("DFT Interpolation")
    axes[3].imshow(img_target_np, cmap="gray")
    axes[3].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")

    plt.show()