'''before starting, make sure to install poppler utils
using this command: brew install poppler
'''

from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi = 300)
    for i, image in enumerate(images):
        image.save(f"{output_folder}/page_{i + 1}.jpg", "JPEG")


# Example usage:
if __name__ == "__main__":
    pdf_path = "/Users/mayankgupta/Desktop/Construct/AutoGraderSystem/Dummy_Data/inputs/IMG20250415224436.pdf" 
    output_folder = "/Users/mayankgupta/Desktop/Construct/AutoGraderSystem/Dummy_Data/output/images"
    convert_pdf_to_images(pdf_path, output_folder)

