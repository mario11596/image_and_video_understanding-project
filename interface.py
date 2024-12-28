from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QWidget, QSplitter
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import cv2
import torch
from PIL import Image
from featureDetection.live_detection import process_video_frame
import rnn_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_mapping = [
            'A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'minus',
                   'N', 'O', 'P', 'parentheses', 'period', 'Q', 'question mark', 'R', 'S', 'Space', 'T', 'U', 'V', 'W',
                   'X', 'Y', 'Z'
            ]

# Interface of the application with setting of the layouts
class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")

        self.main_layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()

        self.splitter = QSplitter(Qt.Horizontal)

        self.camera_label = QLabel("Camera Feed")
        self.splitter.addWidget(self.camera_label)

        self.letter_label = QLabel("Recognized Letter")
        self.splitter.addWidget(self.letter_label)

        self.splitter.setSizes([self.splitter.width() // 2, self.splitter.width() // 2])

        self.top_layout.addWidget(self.splitter)
        self.main_layout.addLayout(self.top_layout)

        self.text_input = QLineEdit()
        self.text_input.setReadOnly(True)

        font = QFont()
        font.setPointSize(14)
        self.text_input.setFont(font)
        self.text_input.setFixedHeight(40)

        self.main_layout.addWidget(self.text_input)

        self.setLayout(self.main_layout)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Camera not found or is unavailable.")
            return

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_camera)
        self.video_timer.start(30)

        self.process_timer = QTimer(self)
        self.process_timer.timeout.connect(self.process_frame)
        self.process_timer.start(1000)

        self.recognized_text = ""
        self.latest_letter = None

        self.model = self.load_model("sign_language_rnn_model.pth")
        self.latest_features = None

    # Load the trained RNN model
    def load_model(self, model_path):
        model = rnn_model.RNNModel().to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    # Show on camera new frame with landmarks
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            frame_to_show, features = process_video_frame(frame)

            self.latest_features = features

            # Convert frame to RGB for display
            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

    def process_frame(self):
        if self.latest_features is not None and self.latest_features.size > 0:

            letter = self.recognize_letter(self.latest_features)
            if letter:
                self.latest_letter = letter
                self.update_recognized_letter(letter)

    # Predict the sign using the model
    def recognize_letter(self, features):
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(feature_tensor)
            _, predicted_class = torch.max(output, 1)
            class_index = predicted_class.item()

            # Check if the index is valid in the class mapping
            if 0 <= class_index < len(class_mapping):
                letter = class_mapping[class_index]
            else:
                letter = None
        return letter

    # Show the sign on the left part of application window
    def update_recognized_letter(self, letter):

        self.recognized_text += letter
        self.text_input.setText(self.recognized_text)

        letter_image = self.get_letter_image(letter)
        if letter_image:
            self.letter_label.setPixmap(QPixmap.fromImage(letter_image))

    # Get the sign image
    def get_letter_image(self, letter):
        try:
            letter_img_path = f"letters/{letter}.png"
            img = Image.open(letter_img_path)
            img = img.convert("RGB")
            qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)

            return qimg

        except Exception as e:
            print(f"Error loading letter image: {e}")
            return None

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = SignLanguageApp()
    window.show()
    app.exec()
