import os
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import vision


def read_images_from_folders(base_folder):
    for root, _, files in os.walk(base_folder):
        folder_name = os.path.basename(root)
        for file_name in files:
            if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
                yield os.path.join(root, file_name), folder_name


def process_hand_landmarker_result(output, label):
    # Extract left and right hand landmarks as flattened arrays
    left_hand_landmarks = [0.0] * 63
    right_hand_landmarks = [0.0] * 63

    for i, handedness in enumerate(output.handedness):
        hand_landmarks = output.hand_landmarks[i]
        flattened_landmarks = [
                                  landmark.x for landmark in hand_landmarks
                              ] + [
                                  landmark.y for landmark in hand_landmarks
                              ] + [
                                  landmark.z for landmark in hand_landmarks
                              ]

        if handedness[0].category_name == "Left":
            left_hand_landmarks = flattened_landmarks
        elif handedness[0].category_name == "Right":
            right_hand_landmarks = flattened_landmarks

    return {
        "label": label,
        "landmarkers_leftHand": np.array(left_hand_landmarks, dtype=np.float32),
        "landmarkers_rightHand": np.array(right_hand_landmarks, dtype=np.float32)
    }


if __name__ == '__main__':
    #TODO: put correct folder for dataset!
    base_folder = '../dataset/'
    model_path = 'hand_landmarker.task'
    splits = ["train", "validation", "train"]

    mp_hands = mp.solutions.hands
    base_options = BaseOptions(model_asset_path=model_path)

    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE)

    with HandLandmarker.create_from_options(options) as landmarker:
        for split in splits:
            print(f"Processing {split} split...")
            split_path = os.path.join(base_folder, split)
            label_to_idx = {label: idx for idx, label in enumerate(sorted(os.listdir(split_path)))}

            features = []
            labels = []

            for image_path, folder_label in read_images_from_folders(split_path):
                pil_image = Image.open(image_path)
                numpy_image = np.array(pil_image)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

                result = landmarker.detect(mp_image)
                #TODO: define where images should be saved
                #output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.npy"
                processed_data = process_hand_landmarker_result(result, folder_label)
                combined_landmarks = np.concatenate(
                    (processed_data["landmarkers_leftHand"], processed_data["landmarkers_rightHand"])
                )
                features.append(combined_landmarks)
                labels.append(label_to_idx[processed_data["label"]])

            features = np.array(features, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)

            np.save(f"{split}_data.npy", features)
            np.save(f"{split}_labels.npy", labels)
            #print(f"Saved data to {output_filename}")