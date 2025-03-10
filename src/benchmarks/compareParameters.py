import argparse as ap 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import tifffile
from os.path import join
from concurrent.futures import ProcessPoolExecutor
from skimage.segmentation import chan_vese
from skimage.transform import rescale
from skimage.morphology import binary_closing, disk
from skimage.filters import gaussian

def get_zstack(file_path:str) -> np.ndarray:
    try:
        z_stack = tifffile.imread(file_path)
    except Exception as e:
        print(str(e))
        exit(1)

    z_stack = z_stack / 255.0
    z_stack = gaussian(z_stack, sigma=1, preserve_range=True)

    # return np.array([rescale(z_stack[i], 0.7, anti_aliasing=True) for i in range(len(z_stack))])
    return z_stack

def process_image(i, image, mu, lambda_1, lambda_2, disk_size):
    """Applies Chan-Vese segmentation to an image and post-processes it."""
    cv = chan_vese(image, mu=mu, lambda1=lambda_1, lambda2=lambda_2,
                   max_num_iter=100, extended_output=True, tol=1e-5)
    segmented_image = np.invert(cv[0]).astype(np.uint8)
    segmented_image = binary_closing(segmented_image, disk(disk_size))
    return i, segmented_image  # Return index and result

def parallel_segmentation(z_stack, mu, lambda_1, lambda_2, disk_size):
    segments = [None] * len(z_stack)  # Preallocate list for order preservation
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, i, z_stack[i], mu, lambda_1, lambda_2, disk_size)
                   for i in range(len(z_stack))]
        
        for future in futures:
            i, segmented_image = future.result()
            segments[i] = segmented_image  # Place result in correct index

    return segments

def generate_data(z_stack:np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    lambda_1 = 1.0 
    lambda_2 = 1.3
    mu = 0.05 # 0.1
    disk_size = 1

    while True:
        print(f"Current params\nl1:\t\t{lambda_1}\nl2:\t\t{lambda_2}\nmu:\t\t{mu}\ndisk_size:\t{disk_size}")
        segments = parallel_segmentation(z_stack, mu, lambda_1, lambda_2, disk_size)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=150)  # Increase size and resolution
        plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.9)  # Adjust spacing

        idx = 0

        img_plot = ax[0].imshow(z_stack[idx], cmap='gray')
        seg_plot = ax[1].imshow(segments[idx], cmap='gray')
        ax[0].set_title("Original Image")
        ax[1].set_title("Segmented Image")

        ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, len(z_stack) - 1, valinit=0, valstep=1)

        def update(val):
            idx = int(slider.val)
            img_plot.set_data(z_stack[idx])
            seg_plot.set_data(segments[idx])
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

        prompt = input("Are the parameters okay? y/n: ")

        if prompt == 'y':
            break

        lambda_1 = float(input(f"Lambda_1 = {lambda_1}: "))
        print()
        lambda_2 = float(input(f"Lambda_2 = {lambda_2}: "))
        print()
        mu = float(input(f"mu = {mu}: "))
        print()
        disk_size = int(input(f"Disk size = {disk_size}: "))
        print()

    return z_stack, segments

def save_results(z_stack:np.ndarray, segments:np.ndarray, output_image:str, output_segments:str, filename:str) -> None:
    try:
        np.save(join(output_image, f"{filename}.npy"), z_stack, allow_pickle=True)
        np.save(join(output_segments, f"{filename}.npy"), segments, allow_pickle=True)
    except Exception as e:
        print(str(e))
    
def main() -> None:
    parser = ap.ArgumentParser(
        "makeDataset.py: program to generate a ground truth given an input z-stack."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input .tif image."
    )

    parser.add_argument(
        "--output_img",
        type=str,
        required=True,
        help="Folder for the output img."
    )

    parser.add_argument(
        "--output_ground",
        type=str,
        required=True,
        help="Folder for the output segmentation."
    )

    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Name to use for image and segmentation"
    )

    args = parser.parse_args()

    z_stack = get_zstack(args.input)

    images, segments = generate_data(z_stack)
    
    save_results(images, segments, args.output_img, args.output_ground, args.output_name)


if __name__ == '__main__':
    main()