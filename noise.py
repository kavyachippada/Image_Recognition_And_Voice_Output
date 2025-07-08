import cv2
import numpy as np

image = cv2.imread('Capture.png', cv2.IMREAD_GRAYSCALE)

np.random.seed(42)  
noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
noise_intensity = 0.6  
noisy_image = cv2.addWeighted(image, 1 - noise_intensity, noise, noise_intensity, 0)

cv2.imwrite('noisy_image.png', noisy_image)
print(f"Moderate XOR-based noise added (intensity = {noise_intensity}). Noisy image saved as 'noisy_image.png'.")