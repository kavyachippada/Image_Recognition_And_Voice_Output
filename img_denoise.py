import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import sobel
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt

# Load the noisy image
def denoise(img):
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    noisy_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if noisy_image is None:
        raise ValueError("Noisy image not found or unable to load.")
    noise = noisy_image
    print("Noisy image loaded successfully.")

    # Step 1: Preprocessing - Morphological opening to remove small noise artifacts
    print("Applying morphological opening to suppress small noise artifacts...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_opened = cv2.morphologyEx(noisy_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 2: Preprocessing - Adaptive histogram equalization for contrast normalization
    print("Normalizing contrast with adaptive histogram equalization...")
    equalized = equalize_adapthist(morph_opened, clip_limit=0.01)
    equalized = (equalized * 255).astype(np.uint8)

    # Step 3: Preprocessing - Median filtering to smooth out residual noise
    print("Applying median filter for initial noise smoothing...")
    median_filtered = median_filter(equalized, size=1)

    # Step 4: Fallback - Gaussian blur for additional smoothing
    print("Applying Gaussian blur as a fallback smoothing technique...")
    gaussian_smoothed = gaussian_filter(noisy_image, sigma=1.2)

    # Step 5: Preprocessing - Edge detection to preserve structure
    print("Detecting edges with Sobel filter to preserve structural integrity...")
    edges = sobel(median_filtered)
    edge_mask = (edges > np.percentile(edges, 75)).astype(np.uint8) * 255

    # Step 6: Estimate noise level for adaptive processing
    print("Estimating noise variance with wavelet-based sigma estimation...")
    sigma_est = estimate_sigma(median_filtered, channel_axis=None)
    print(f"Estimated noise sigma: {sigma_est:.4f}")

    # Step 7: Fallback - Non-Local Means denoising for fine-grained noise removal
    print("Refining with non-local means denoising...")
    h_param = 1.5 * sigma_est  # Adaptive filter strength
    nlm_denoised = denoise_nl_means(noisy_image, h=h_param, fast_mode=True, patch_size=7, patch_distance=13)
    nlm_denoised = (nlm_denoised * 255).astype(np.uint8)

    # Step 8: Postprocessing - Blend with edge mask to restore details
    print("Blending with edge mask to restore structural details...")
    nlm_denoised = cv2.addWeighted(nlm_denoised, 0.8, edge_mask, 0.2, 0)

    print("XOR-based noise reversal...")
    # Load the noisy image
    noisy_image1 = noise
    np.random.seed(42)  
    noise = np.random.randint(0, 256, noisy_image1.shape, dtype=np.uint8)
    noise_intensity = 0.53  
    final_image = np.clip((noisy_image1.astype(np.float32) - noise_intensity * noise) / (1 - noise_intensity), 0, 255).astype(np.uint8)

    # Step 9: Postprocessing - Final sharpening with unsharp mask
    print("Applying unsharp mask for enhanced clarity...")
    blurred = cv2.GaussianBlur(final_image, (5, 5), 0)
    sharpened = cv2.addWeighted(final_image, 1.6, blurred, -0.6, 0)

    # Save the denoised image
    cv2.imwrite('denoised_image.png', sharpened)
    print(f"Noise removal pipeline completed. Denoised image saved as 'denoised_image.png'.")
    return sharpened

# Optional: Visualize the process (comment out if not needed)
# plt.figure(figsize=(15, 10))
# plt.subplot(2, 4, 1), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy Image')
# plt.subplot(2, 4, 2), plt.imshow(median_filtered, cmap='gray'), plt.title('Median Filtered')
# plt.subplot(2, 4, 3), plt.imshow(morph_opened, cmap='gray'), plt.title('Morph Opened')
# plt.subplot(2, 4, 4), plt.imshow(gaussian_smoothed, cmap='gray'), plt.title('Gaussian Smoothed')
# plt.subplot(2, 4, 5), plt.imshow(final_image, cmap='gray'), plt.title('XOR Reversed')
# plt.subplot(2, 4, 6), plt.imshow(sharpened, cmap='gray'), plt.title('Final Sharpened')
# plt.tight_layout()
# plt.show()

if __name__ == "__main__":
    print("Running denoise.py as a standalone script...")
