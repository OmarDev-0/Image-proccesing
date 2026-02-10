import cv2
import numpy as np
import matplotlib.pyplot as plt

# Functions

def sample_image(image, factor):
    
    h, w = image.shape
    sampled = cv2.resize(
        image,
        (w // factor, h // factor),
        interpolation=cv2.INTER_NEAREST
    )
    return sampled


def quantize_image(image, levels):
    
    step = 256 // levels
    quantized = np.floor(image / step) * step
    return quantized.astype(np.uint8)


def plot_images(original, sampled, quantized):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sampled, cmap='gray')
    plt.title("Sampled Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(quantized, cmap='gray')
    plt.title("Quantized Image")
    plt.axis("off")

    plt.show()

# Loading images

# Parameters
sampling_factor = 8
quantization_levels = 16

# Load images in grayscale (already are)
img1 = cv2.imread("Circle.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Square.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Could not load images. Check file names.")
    exit()

# Resize to same size
img1 = cv2.resize(img1, (256, 256))
img2 = cv2.resize(img2, (256, 256))

# Task 1

sampled_image = sample_image(img1, sampling_factor)
quantized_image = quantize_image(img1, quantization_levels)

plot_images(img1, sampled_image, quantized_image)

# Task 2

# Subtract two images
subtraction = cv2.subtract(img1, img2)
plt.imshow(subtraction, cmap='gray', vmin = 0, vmax = 255)
plt.title("Subtraction (Image1 - Image2)")
plt.axis("off")
plt.show()

# Adding a constant
added_constant = cv2.add(img1, 175)
plt.imshow(added_constant, cmap='gray', vmin = 0, vmax = 255)
plt.title("Add Constant 175")
plt.axis("off")
plt.show()



# Symmetric Difference (XOR)

symmetric_difference = cv2.bitwise_xor(img1, img2)
plt.imshow(symmetric_difference, cmap='gray', vmin=0, vmax=255)
plt.title("Symmetric Difference (XOR)")
plt.axis("off")
plt.show()


# Intersection (AND)

intersection = cv2.bitwise_and(img1, img2)
plt.imshow(intersection, cmap='gray', vmin=0, vmax=255)
plt.title("Intersection (AND)")
plt.axis("off")
plt.show()



