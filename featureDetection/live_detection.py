import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.multi_hand_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        for landmark in hand_landmarks.landmark:
            hand_landmarks_proto.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z)

        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

    return annotated_image

def process_hand_landmarker_result(output):
    # Extract left and right hand landmarks as flattened arrays
    left_hand_landmarks = [0.0] * 63
    right_hand_landmarks = [0.0] * 63

    for i, handedness in enumerate(output.multi_handedness):
        hand_landmarks = output.multi_hand_landmarks[i]
        flattened_landmarks = [
                                  landmark.x for landmark in hand_landmarks.landmark
                              ] + [
                                  landmark.y for landmark in hand_landmarks.landmark
                              ] + [
                                  landmark.z for landmark in hand_landmarks.landmark
                              ]

        print(flattened_landmarks)
        if handedness.classification[0].label == "Left":
            right_hand_landmarks = flattened_landmarks
        elif handedness.classification[0].label == "Right":
            left_hand_landmarks = flattened_landmarks

    return {
        "landmarkers_leftHand": np.array(left_hand_landmarks, dtype=np.float32),
        "landmarkers_rightHand": np.array(right_hand_landmarks, dtype=np.float32)
    }

def process_video_frame(frame):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        features = []

        if results.multi_hand_landmarks:
            annotated_image = draw_landmarks_on_image(frame_rgb, results)

            processed_data = process_hand_landmarker_result(results)
            combined_landmarks = np.concatenate(
                (processed_data["landmarkers_leftHand"], processed_data["landmarkers_rightHand"])
            )
            features = combined_landmarks
            features = np.array(features, dtype=np.float32)

            return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR), features
        else:
            return frame, None