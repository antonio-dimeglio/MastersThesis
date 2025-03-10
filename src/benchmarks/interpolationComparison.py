import tifffile
import numpy as np

def linear_interpolation(A:np.ndarray, B:np.ndarray, alpha=0.5) -> np.ndarray:
    # Given two 2D slices, it returns the linear interpolation between the two
    return (1-alpha)*A+ alpha * B

import numpy as np
import scipy.interpolate as interp

def cubicspline_interpolation(A: np.ndarray, B: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Given two 2D slices, interpolate between them using cubic splines.

    Parameters:
        A (np.ndarray): First 2D slice.
        B (np.ndarray): Second 2D slice.
        alpha (float): Interpolation factor (0 = A, 1 = B, 0.5 = midpoint).

    Returns:
        np.ndarray: Interpolated 2D slice.
    """
    # Define the known slices and the query point
    z_known = np.array([0, 1])  # A corresponds to 0, B corresponds to 1
    z_interp = np.array([alpha])  # Query interpolation point

    # Interpolation along rows first
    interpolated_rows = np.zeros_like(A, dtype=np.float64)
    for i in range(A.shape[0]):
        spline = interp.CubicSpline(z_known, np.vstack([A[i, :], B[i, :]]), axis=0)
        interpolated_rows[i, :] = spline(z_interp)

    # Interpolation along columns
    interpolated_result = np.zeros_like(A, dtype=np.float64)
    for j in range(A.shape[1]):
        spline = interp.CubicSpline(z_known, np.vstack([A[:, j], B[:, j]]), axis=0)
        interpolated_result[:, j] = spline(z_interp)

    return interpolated_result


def fft_interpolation(A:np.ndarray, B:np.ndarray, alpha) -> np.ndarray:
    # Given two 2D slices, it returns the FFT linear interpolation between the two
    FA = np.fft.fft2(A)
    FB = np.fft.fft2(B)
    F_interp = (1 - alpha) * FA + alpha * FB
    return np.fft.ifft2(F_interp).real
import scipy.interpolate as interp
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

def evaluate_interpolation(gt, interp_img):
    """ Compute various metrics to compare interpolation with ground truth """
    mse = mean_squared_error(gt, interp_img)
    psnr = peak_signal_noise_ratio(gt, interp_img, data_range=gt.max() - gt.min())
    ssim = structural_similarity(gt, interp_img, data_range=gt.max() - gt.min())
    ncc = np.corrcoef(gt.ravel(), interp_img.ravel())[0, 1]  # Normalized Cross-Correlation
    
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "NCC": ncc}

# Load images
img1 = tifffile.imread("/mnt/d/d1/AB-TGF/15_D6.tif")[-1, :, :]
img3 = tifffile.imread("/mnt/d/d1/AB-TGF/15_D6.tif")[-3, :, :]
img2 = tifffile.imread("/mnt/d/d1/AB-TGF/15_D6.tif")[-2, :, :]  # Ground truth

# Interpolations
img2_linear = linear_interpolation(img1, img3, alpha=0.5)
img2_cubic = cubicspline_interpolation(img1, img3, alpha=0.5)
img2_fft = fft_interpolation(img1, img3, alpha=0.5)

# Evaluate
metrics_linear = evaluate_interpolation(img2, img2_linear)
metrics_cubic = evaluate_interpolation(img2, img2_cubic)
metrics_fft = evaluate_interpolation(img2, img2_fft)

# Print results
print("Linear Interpolation:", metrics_linear)
print("Cubic Spline Interpolation:", metrics_cubic)
print("FFT Interpolation:", metrics_fft)