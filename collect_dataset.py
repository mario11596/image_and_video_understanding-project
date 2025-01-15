import os
import cv2

# Create dataset from letter A to Z
def create_custom_dataset():
    # put here the name of cusom folder names
    class_names = ['A', 'B', 'C', 'comma', 'D', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'minus',
                   'N', 'O', 'P', 'period', 'Q', 'question mark', 'R', 'S', 'space', 'T', 'U', 'V', 'W',
                   'X', 'Y', 'Z']

    # path for custom repository
    DATA_DIR = './custom_dataset'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    number_of_classes = len(class_names)

    # Number of images per class
    dataset_size = 100

    cap = cv2.VideoCapture(0)

    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, class_names[j])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print('Collecting data for class {}'.format(class_names[j]))

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.putText(frame, None,(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Press SPACE to save the image
                image_name = '{}{}.jpg'.format(class_names[j], counter)
                cv2.imwrite(os.path.join(class_dir, image_name), frame)
                print(f"Saved {image_name}")

                counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    create_custom_dataset()