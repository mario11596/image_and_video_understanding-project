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

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                annotated_image = draw_landmarks_on_image(frame_rgb, results)
                frame_to_show = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            else:
                frame_to_show = frame

            # !! Fixed: also show webcam when no hands are visible
            cv2.imshow("Webcam Feed with Hand Landmarks", frame_to_show)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
