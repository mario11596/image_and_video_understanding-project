import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import json


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def read_image():
    model_path = 'hand_landmarker.task'

    base_options = BaseOptions(model_asset_path=model_path)

    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE)

    pil_image = Image.open('images/Comma1.jpg')
    numpy_image = np.array(pil_image)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)
        output_path = 'annotated_image.jpg'
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Annotated image saved to {output_path}")
        process_hand_landmarker_result(result, "Comma1")

    print(result)

def process_hand_landmarker_result(output, img_name):
    data = {}

    handedness = []
    for category in output.handedness:
        handedness.append({
            "score": category[0].score,
            "display_name": category[0].display_name,
            "category_name": category[0].category_name
        })

    hand_landmarks = []
    for hand in output.hand_landmarks:
        hand_landmarks.append({
            "NormalizedLandmark": [
                {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                for landmark in hand
            ]
        })

    world_landmarks = []
    for world_hand in output.hand_world_landmarks:
        world_landmarks.append({
            "Landmark": [
                {"x": landmark.x, "y": landmark.y, "z": landmark.z}
                for landmark in world_hand
            ]
        })

    data[img_name] = [
        {
            "handedness": handedness,
            "hand_landmarks": hand_landmarks
        }
    ]
    data["world_markers"] = world_landmarks

    with open('output.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Data written to output.json")


if __name__ == '__main__':
    read_image()