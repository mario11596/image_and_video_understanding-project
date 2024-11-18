import os
import cv2
from pathlib import Path

input_folder = "./custom_dataset"   
output_folder = "./p_custom_dataset" 

def process_images(input_folder, output_folder, image_size=(200, 200)):

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    output_path.mkdir(parents=True, exist_ok=True)

    for subdir, _, files in os.walk(input_path):
        relative_path = Path(subdir).relative_to(input_path)
        target_folder = output_path / relative_path
        target_folder.mkdir(parents=True, exist_ok=True)

        for file in files:
            input_file_path = Path(subdir) / file
            image = cv2.imread(str(input_file_path))

            # Grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize 
            normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

            # Resize 
            resized_image = cv2.resize(normalized_image, image_size)

            # Save 
            output_file_path = target_folder / file
            cv2.imwrite(str(output_file_path), resized_image)

            print(f"{output_file_path}")


process_images(input_folder, output_folder)
