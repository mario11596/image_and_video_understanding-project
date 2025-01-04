import os
import shutil
from sklearn.model_selection import train_test_split

folder_names = ['A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'minus', 'N', 'O', 'P', 'period', 'Q', 'question mark', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z']

# Paths for each part
train_directory = r'.\dataset\train'
validation_directory = r'.\dataset\validation'
test_directory = r'.\dataset\test'
source_base = r'.\custom_dataset'

test_size = 0.2


# Split dataset on three parts: training/validation/testing
def split_dataset():
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(validation_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    for folder_name in folder_names:
        source_folder = os.path.join(source_base, folder_name)
        train_folder = os.path.join(train_directory, folder_name)
        validation_folder = os.path.join(validation_directory, folder_name)
        test_folder = os.path.join(test_directory, folder_name)

        images = [os.path.join(source_folder, img) for img in os.listdir(source_folder) if
                  img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        train_images, test_images = train_test_split(images, test_size=test_size, random_state=302)

        train_images, validation_images = train_test_split(train_images, test_size=test_size, random_state=302)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(validation_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        for img in train_images:
            shutil.copy(img, train_folder)

        for img in validation_images:
            shutil.copy(img, validation_folder)

        for img in test_images:
            shutil.copy(img, test_folder)


if __name__ == '__main__':
    split_dataset()
