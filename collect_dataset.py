import os
import cv2


def create_custom_dataset():
    # put here the name of cusom folder names
    class_names = ['parentheses', 'question mark']

    # path for custom repository
    DATA_DIR = './custom_dataset'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    number_of_classes = len(class_names)
    dataset_size = 50

    cap = cv2.VideoCapture(0)

    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, class_names[j])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print('Collecting data for class {}'.format(class_names[j]))

        done = False
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, None, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            image_name = '{}{}.jpg'.format(class_names[j], counter)
            cv2.imwrite(os.path.join(class_dir, image_name), frame)

            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    create_custom_dataset()