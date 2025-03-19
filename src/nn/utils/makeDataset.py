import os
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.segmentation import chan_vese
from skimage.morphology import binary_closing, disk
from tifffile import imread
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def process_slice(
        img: np.ndarray,
        sigma: float=3,
        mu: float=0.05,
        lambda1: float=1.0,
        lambda2: float=1.3,
        tol: float=1e-5,
        iterations: int=100,
        disk_size: int=1) -> np.ndarray:    
    
    img = img / 255.0
    img = rescale_intensity(img)
    img = gaussian(img, sigma=sigma, preserve_range=True)

    cv = chan_vese(img, mu=mu, lambda1=lambda1, lambda2=lambda2, tol=tol, 
                max_num_iter=iterations, dt=0.5, init_level_set="checkerboard", extended_output=True)

    segmented_image = np.invert(cv[0]) + 0  
    segmented_image = binary_closing(segmented_image, disk(disk_size))
    
    return segmented_image.astype(np.uint8)*255

def process_slice_and_save(i, img, file, dataset_images, dataset_ground):
    segmentation = process_slice(img)

    # Save the original image and the segmentation
    Image.fromarray(img).save(os.path.join(
        dataset_images,
        f"{file}_{i}.png"
    ))

    Image.fromarray(segmentation).save(os.path.join(
        dataset_ground,
        f"{file}_{i}.png"
    ))

def process_file(filepath, z_stack, dataset_images, dataset_ground):
    with ProcessPoolExecutor() as executor:
        # For each slice in the z-stack, submit the slice processing task
        for i in range(z_stack.shape[0]):
            print(f"Processing slice {i+1}/{z_stack.shape[0]}")
            img = z_stack[i]
            executor.submit(process_slice_and_save, i, img, filepath, dataset_images, dataset_ground)
            
def main():
    src_folder = "/mnt/d/d_merged12"
    dataset_images = "/home/quantum/MScThesisCode/dataset/images"
    dataset_ground = "/home/quantum/MScThesisCode/dataset/ground"
    categories = os.listdir(src_folder)

    for category in categories:
        print(f"Processing {category}.")
        folder = os.path.join(src_folder, category)
        files = os.listdir(folder)
        
        for file in files:
            print(f"Processing {file}.")
            filepath = os.path.join(folder, file)
            z_stack = imread(filepath)

            # Process each file with slice-level parallelization
            process_file(file, z_stack, dataset_images, dataset_ground)

if __name__ == '__main__':
    main()
