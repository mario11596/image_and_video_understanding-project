import os
import kagglehub
import shutil
from sklearn.model_selection import train_test_split

folder_names = ['A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'minus', 'N', 'nothing', 'O', 'P', 'parentheses', 'period', 'Q', 'question mark', 'R', 'S', 'space',
                'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def download_kaggle_dataset():
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")

    print("Path to dataset files:", path)

    return path


def copy_dataset(path_kaggle):
    destination_path = os.getcwd()
    if os.path.isdir(path_kaggle):
        shutil.copytree(path_kaggle, os.path.join(destination_path, os.path.basename(path_kaggle)))
        print(f"Folder copied to {destination_path}")
    else:
        print("Source path does not exist.")


def combine_datasets():
    source_path = r'.\custom_dataset'
    destination_path = r'.\1\asl_alphabet_train\asl_alphabet_train'

    if os.path.exists(source_path) and os.path.exists(destination_path):
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            if os.path.isdir(item_path):
                shutil.copytree(item_path, os.path.join(destination_path, item), dirs_exist_ok=True)
            else:
                shutil.copy2(item_path, destination_path)
    else:
        print("Source or destination path does not exist.")


def split_dataset():
    train_directory = r'.\dataset\train'
    validation_directory = r'.\dataset\validation'
    source_base = r'.\1\asl_alphabet_train\asl_alphabet_train'

    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(validation_directory, exist_ok=True)

    for folder_name in folder_names:
        source_folder = os.path.join(source_base, folder_name)
        train_folder = os.path.join(train_directory, folder_name)
        validation_folder = os.path.join(validation_directory, folder_name)

        images = [os.path.join(source_folder, img) for img in os.listdir(source_folder) if
                  img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        train_images, validation_images = train_test_split(images, train_size=0.8, random_state=302)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(validation_folder, exist_ok=True)

        for img in train_images:
            shutil.copy(img, train_folder)

        for img in validation_images:
            shutil.copy(img, validation_folder)


if __name__ == '__main__':
    path_kaggle = download_kaggle_dataset()
    copy_dataset(path_kaggle)
    combine_datasets()
    split_dataset()
