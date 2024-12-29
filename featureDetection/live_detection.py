import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model_path = 'featureDetection/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

current_landmarks = None

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_landmarks
    if result.hand_landmarks:
        current_landmarks = result.hand_landmarks
    else:
        current_landmarks = None  # Clear current landmarks if no hands detected

options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=1
    )

landmarker = HandLandmarker.create_from_options(options)

def draw_landmarks_on_frame(frame, landmarks):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    for hand_landmarks in landmarks:
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            points.append((x, y))
            cv2.circle(frame, (x, y), 8, (0, 178, 0), -1)

        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (86, 22, 217), 3)

def process_hand_landmarker_result(output):
    left_hand_landmarks = [0.0] * 63
    right_hand_landmarks = [0.0] * 63

    for hand_index, hand_landmarks in enumerate(output):
        flattened_landmarks = (
            [landmark.x for landmark in hand_landmarks] +
            [landmark.y for landmark in hand_landmarks] +
            [landmark.z for landmark in hand_landmarks]
        )

        if hand_index == 1:  #Left
            left_hand_landmarks = flattened_landmarks
        elif hand_index == 0:  #Right
            right_hand_landmarks = flattened_landmarks

    return {
        "landmarkers_leftHand": np.array(left_hand_landmarks, dtype=np.float32),
        "landmarkers_rightHand": np.array(right_hand_landmarks, dtype=np.float32)
    }

def process_video_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    if current_landmarks:
        draw_landmarks_on_frame(frame, current_landmarks)
        processed_data = process_hand_landmarker_result(current_landmarks)
        combined_landmarks = np.concatenate(
            (processed_data["landmarkers_leftHand"], processed_data["landmarkers_rightHand"])
        )
        features = np.array(combined_landmarks, dtype=np.float32)
        return frame, features

    return frame, [] #nothing detected