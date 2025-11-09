import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def preprocess_image(image_path, output_path, debug=False):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Apply sharpening to the enhanced image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Apply adaptive thresholding with fine-tuned parameters
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, blockSize=15, C=5)

    # Save the preprocessed image
    cv2.imwrite(output_path, binary)

    if debug:
        # Optional: Display the images
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 5, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 5, 2)
        plt.title('Grayscale Image')
        plt.imshow(gray, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 5, 3)
        plt.title('Denoised Image')
        plt.imshow(denoised, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 5, 4)
        plt.title('Enhanced & Sharpened Image')
        plt.imshow(sharpened, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 5, 5)
        plt.title('Binary Image')
        plt.imshow(binary, cmap='gray')
        plt.axis('off')

        plt.show()


if __name__ == "__main__":
    os.makedirs('../Dummy_Data/output/preprocessed_images', exist_ok=True)
    input_image_folder = '../Dummy_Data/output/images'
    output_image_folder = '../Dummy_Data/output/preprocessed_images'

    for image_file in os.listdir(input_image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_image_path = os.path.join(input_image_folder, image_file)
            output_image_path = os.path.join(output_image_folder, image_file)
            preprocess_image(input_image_path, output_image_path, debug=True)