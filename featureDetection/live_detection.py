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
        current_landmarks = result.hand_landmarks #set current_landmarks to detected hand
    else:
        current_landmarks = None  #if no hands detected set to none

options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM, #Mode = Live stream since real time capturing of features
        result_callback=print_result,
        num_hands=1 #We only want to detect one hand
    )

landmarker = HandLandmarker.create_from_options(options) #Initialize media pipe hand marker detection

def draw_landmarks_on_frame(frame, landmarks):
    #Features on hand
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    #visualize the features as connected circles on the live video
    for hand_landmarks in landmarks:
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            points.append((x, y))
            cv2.circle(frame, (x, y), 8, (245,216,90), -1)

        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (115,134,120), 3)

def process_hand_landmarker_result(output):
    hand_landmarks_detection = [0.0] * 63
    #extract x,y,z value for each feature
    if output is not None:
        for hand_index, hand_landmarks in enumerate(output):
            flattened_landmarks = (
                [landmark.x for landmark in hand_landmarks] +
                [landmark.y for landmark in hand_landmarks] +
                [landmark.z for landmark in hand_landmarks]
            )

            hand_landmarks_detection = flattened_landmarks

    return {
        "hand_landmarks_detection": np.array(hand_landmarks_detection, dtype=np.float32)
    }

def process_video_frame(frame):
    #read in video frame and compute features
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    landmarker.detect_async(mp_image, timestamp_ms)

    if current_landmarks:
        draw_landmarks_on_frame(frame, current_landmarks)
        processed_data = process_hand_landmarker_result(current_landmarks)
        combined_landmarks = processed_data["hand_landmarks_detection"]

        features = np.array(combined_landmarks, dtype=np.float32)
        return frame, features

    return frame, [] #nothing was detected