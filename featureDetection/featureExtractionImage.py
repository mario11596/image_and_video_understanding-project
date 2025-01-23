import os
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python import vision


#Read in all images from base_folder
def read_images_from_folders(base_folder):
    for root, _, files in os.walk(base_folder):
        folder_name = os.path.basename(root)
        for file_name in files:
            if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
                yield os.path.join(root, file_name), folder_name


def process_hand_landmarker_result(output, label):
    #Extract left and right hand landmarks as flattened arrays
    hand_landmarks_detection = [0.0] * 63 #*63 because 21 features per hand times x,y,z value

    #for each detected hand retrieve marker x,y,z values
    for i, handedness in enumerate(output.handedness):
        hand_landmarks = output.hand_landmarks[i]
        flattened_landmarks = [
                                  landmark.x for landmark in hand_landmarks
                              ] + [
                                  landmark.y for landmark in hand_landmarks
                              ] + [
                                  landmark.z for landmark in hand_landmarks
                              ]

        #handedness = if right or left hand -> since we detect only one hand does not matter for us
        hand_landmarks_detection = flattened_landmarks #save landmarks (features)

    return {
        "label": label,
        "hand_landmarks_detection": np.array(hand_landmarks_detection, dtype=np.float32),
    }


if __name__ == '__main__':
    base_folder = '../dataset/' #TODO: change path, if your images are stored somewhere else
    model_path = 'hand_landmarker.task' #Mediapipe model
    splits = ["train", "validation", "test"] #Different folders

    mp_hands = mp.solutions.hands
    base_options = BaseOptions(model_asset_path=model_path) #Options for Mediapipe

    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE) #Use running mode IMAGE, since we want to extract features from images
        #not live video

    #Loop through all folders (test, train, validation) and extract features from images
    with HandLandmarker.create_from_options(options) as landmarker:
        for split in splits:
            print(f"Processing {split} split...")
            split_path = os.path.join(base_folder, split)
            label_to_idx = {label: idx for idx, label in enumerate(os.listdir(split_path))}

            features = []
            labels = []

            for image_path, folder_label in read_images_from_folders(split_path):
                #print(folder_label)
                pil_image = Image.open(image_path)
                numpy_image = np.array(pil_image)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

                result = landmarker.detect(mp_image) #Let mediapipe detect features

                processed_data = process_hand_landmarker_result(result, folder_label) #store result and labels
                combined_landmarks = processed_data["hand_landmarks_detection"]

                features.append(combined_landmarks)
                labels.append(label_to_idx[processed_data["label"]])


            features = np.array(features, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)

            #Store all features in .npy files to be used for model
            np.save(f"{split}_data.npy", features)
            np.save(f"{split}_labels.npy", labels)