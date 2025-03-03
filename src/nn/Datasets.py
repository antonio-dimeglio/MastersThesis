import os
from PIL import Image
from torch.utils.data import Dataset

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


class PatchCollagenDataset(Dataset):
    def __init__(self, image_dir, ground_dir, device, patch_size, transform=None):
        super().__init__()

        self.image_dir = image_dir 
        self.ground_dir = ground_dir
        self.device = device 
        self.patch_size = patch_size
        self.transform = transform
        self.image_files = sorted(os.listdir(self.image_dir))  # List all image filename

        self.image_patches = []
        self.ground_patches = []


        # For each image divide it into patches
        # each patch has to be of size (patch_size, patch_size) (the images are greyscale png)
        # the blocks are not supposed to be overlapping, so the stride is gonna match the patch size
        # if the last block size cannot fit the patch size, add enough padding to make up for it.

        # The same must be done for the ground (we want the ground patches to correspond to that of the
        # image they are taken from). The images are also pngs, but their values are between 0 and 1.
        # Round the patches for the ground like this (ground > 0.5).float() 