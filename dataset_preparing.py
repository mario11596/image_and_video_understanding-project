import os
import shutil
from sklearn.model_selection import train_test_split

folder_names = ['A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'minus',
                   'N', 'O', 'P', 'parentheses', 'period', 'Q', 'question mark', 'R', 'S', 'Space', 'T', 'U', 'V', 'W',
                   'X', 'Y', 'Z']

# Paths for each part
train_directory = r'.\dataset1\train'
validation_directory = r'.\dataset1\validation'
test_directory = r'.\dataset1\test'
source_base = r'.\custom_dataset1'

train_size = 0.8


# This method split dataset on three parts: training/validation/testing
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

        train_images, test_images = train_test_split(images, train_size=train_size, random_state=302)

        train_images, validation_images = train_test_split(train_images, train_size=train_size, random_state=302)

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
