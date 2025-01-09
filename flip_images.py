import os
from PIL import Image, ImageOps
import re



import os
from PIL import Image, ImageOps

def flip_images(main_folder):
    suffix = '_f'
    try:
        # Walk through all subdirectories in the main folder
        for root, dirs, files in os.walk(main_folder):
            for file in files:
                # Check if the file is an image (by extension)
                if file.lower().endswith(('.jpg')):
                    print("image", file)
                    file_path = os.path.join(root, file)  # Full path to the image
                    try:
                        # Open the image
                        with Image.open(file_path) as img:
                            # Flip the image horizontally
                            flipped_img = ImageOps.mirror(img)
                            # Create a new filename for the flipped image
                            file_name, file_extension = os.path.splitext(file)
                            flipped_file_name = f"{file_name}{suffix}{file_extension}"
                            flipped_file_path = os.path.join(root, flipped_file_name)
                            # Save the flipped image
                            flipped_img.save(flipped_file_path)
                            print(f"Flipped and saved: {flipped_file_path}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
        print("All images have been processed.")
    except Exception as e:
        print(f"An error occurred: {e}")




# Specify the path to the main folder
main_folder_path = r'C:\Users\agued\OneDrive - Universidad Carlos III de Madrid\TUGraz\Image and Video Recognition\Project\Git6\image_and_video_understanding-project-1\custom_dataset1'

# Call the function to process the images
if __name__ == "__main__":
    flip_images(main_folder_path)