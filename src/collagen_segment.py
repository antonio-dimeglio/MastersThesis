from nn.UNet import UNet
import torch as th
import numpy as np
import tifffile
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_opening, disk

def load_stack(path: str, transform):
    """Load images from a TIFF stack one by one to reduce memory usage."""
    stack = tifffile.imread(path)

    for i in range(stack.shape[0]):
        image = stack[i, :, :]
        image = np.astype(image, np.float32)
        yield transform(image) 


def segment_stack(model: UNet, stack_path: str, transform, device):
    """Process and segment the stack one image at a time."""
    segmented_images = []

    for image in load_stack(stack_path, transform):
        image = image.unsqueeze(0).to(device)  # Add batch and channel dims
        with th.no_grad():
            segment = model(image)
            segment = (segment > 0.5).long().cpu().numpy()[0, 0]  # Convert to numpy
        segment = binary_opening(segment, disk(5))
        segmented_images.append(segment)

    return np.stack(segmented_images, axis=0)  # Stack segmented images

def segment_stack_given_stack(model: UNet, stack: np.ndarray, transform, device):
    """Process and segment the stack one image at a time."""
    segmented_images = []

    for image in stack:
        image = transform(image)
        image = image.unsqueeze(0).to(device)  # Add batch and channel dims
        with th.no_grad():
            segment = model(image)
            segment = (segment > 0.5).long().cpu().numpy()[0, 0]  # Convert to numpy
        segment = binary_opening(segment, disk(5))
        segmented_images.append(segment)

    return np.stack(segmented_images, axis=0)  # Stack segmented images

def get_holes(stack: np.array) -> list[float]:
    """Measures pores area in 2D."""
    results = []
    for i in range(stack.shape[0]):
        img = stack[i, :, :]
        labeled_area = label(img)
        regions = regionprops(labeled_area)

        results.extend([reg.area for reg in regions])
    return results

def get_pores(volume:np.ndarray) -> list[float]:
    """Measures pores volume in 3D stack."""
    results = []
    labeled_volume = label(volume)
    regions = regionprops(labeled_volume)

    for reg in regions:
        results.append(reg.area)
        
    return results 