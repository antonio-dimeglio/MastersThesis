import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import tifffile as tf
from skimage import filters, measure
import trackpy as tp
import pandas as pd

# Read your image stack and normalize
stack = tf.imread("/mnt/d/MAX_C2-spheroid_72h_timelapse_effective.tif")
stacked_segm = []
labeled_stack = []

# Store labeled cells for each time slice
for i in range(stack.shape[0]):
    img = stack[i, :, :]  # Assume that '1' is the channel of interest
    threshold = filters.threshold_otsu(img)  # Get threshold using Otsu's method

    # Segment the image based on the threshold
    segmented_img = img > threshold
    stacked_segm.append(segmented_img)
    
    # Label the segmented regions (cells)
    labeled_img = measure.label(segmented_img)
    labeled_stack.append(labeled_img)

# Convert the list of segmentations into a 3D NumPy array
stacked_segm = np.stack(stacked_segm, axis=0)
labeled_stack = np.stack(labeled_stack, axis=0)

# Define a function to get cell centroids (positions)
def get_cell_positions(labeled_frame, frame_number):
    # Extract positions of cells (centroids)
    props = measure.regionprops(labeled_frame)
    positions = []
    for i, p in enumerate(props):
        # Append the (x, y, frame number) for each cell
        positions.append({'x': p.centroid[1], 'y': p.centroid[0], 'frame': frame_number, 'particle': i})
    return positions

# Initialize a list to hold all the frames (as pandas DataFrames)
all_frames = []

for i in range(stack.shape[0]):  # Loop through all the time slices
    frame_positions = get_cell_positions(labeled_stack[i], i)  # Get positions and frame number
    frame_df = pd.DataFrame(frame_positions)  # Convert to DataFrame
    all_frames.append(frame_df)

# Concatenate the list of DataFrames into one DataFrame
tracking_data = pd.concat(all_frames, ignore_index=True)

# Track cells across time using Trackpy
tracked_cells = tp.link(tracking_data, search_range=5)
tracked_cells = tracked_cells.drop_duplicates(subset=['frame', 'particle'])


# Compute displacement
tracked_cells['displacement'] = np.sqrt(tracked_cells.groupby('particle')['x'].diff()**2 +
                                        tracked_cells.groupby('particle')['y'].diff()**2)

# Compute instantaneous speed
tracked_cells['speed'] = tracked_cells['displacement'] / tracked_cells.groupby('particle')['frame'].diff()

# Compute velocity components
tracked_cells['vx'] = tracked_cells.groupby('particle')['x'].diff()
tracked_cells['vy'] = tracked_cells.groupby('particle')['y'].diff()

# Compute direction change (turning angle)
tracked_cells['angle'] = np.arctan2(tracked_cells['vy'], tracked_cells['vx'])
tracked_cells['angle_change'] = tracked_cells.groupby('particle')['angle'].diff()

# Compute total distance traveled by each cell
tracked_cells['total_distance'] = tracked_cells.groupby('particle')['displacement'].cumsum()

# Compute total displacement (net movement)
total_displacement = tracked_cells.groupby('particle').agg({'x': ['first', 'last'], 'y': ['first', 'last']})
total_displacement['net_displacement'] = np.sqrt((total_displacement[('x', 'last')] - total_displacement[('x', 'first')])**2 + 
                                                 (total_displacement[('y', 'last')] - total_displacement[('y', 'first')])**2)

# Compute mean speed per cell
mean_speed = tracked_cells.groupby('particle')['speed'].mean()

# Compute mean directionality (net displacement / total distance traveled)
directionality = total_displacement['net_displacement'] / tracked_cells.groupby('particle')['total_distance'].last()

# Compute cell persistence (how many frames each cell is tracked)
persistence = tracked_cells.groupby('particle')['frame'].count()

# Compute number of cells per frame
cell_counts = tracked_cells.groupby('frame')['particle'].nunique()

# Print key metrics
print("Tracking Summary:")
print(f"Total cells tracked: {tracked_cells['particle'].nunique()}")
print(f"Mean displacement: {total_displacement['net_displacement'].mean():.2f} pixels")
print(f"Mean speed: {mean_speed.mean():.2f} pixels/frame")
print(f"Mean directionality: {directionality.mean():.2f}")
print(f"Mean persistence (frames per cell): {persistence.mean():.1f}")
print(f"Total number of frames: {tracked_cells['frame'].max() + 1}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot trajectories
tp.plot_traj(tracked_cells, ax=axes[0])
axes[0].set_title("Cell Trajectories")

axes[1].hist(mean_speed, bins=20, edgecolor="black")
axes[1].set_xlabel("Mean Speed (pixels/frame)")
axes[1].set_ylabel("Number of Cells")
axes[1].set_title("Speed Distribution")

# Plot number of tracked cells over time
axes[2].plot(cell_counts, marker="o")
axes[2].set_xlabel("Frame")
axes[2].set_ylabel("Number of Tracked Cells")
axes[2].set_title("Antibody Count Over Time")

plt.tight_layout()
plt.show()

