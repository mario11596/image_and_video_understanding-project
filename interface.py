from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QWidget, QSplitter
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import cv2
import torch
from PIL import Image
from featureDetection.live_detection import process_video_frame
import rnn_model
import actual_cnn
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(10)

        self.recognized_text = ""

        self.model = self.load_model("sign_language_cnn_model.pth")

    def load_model(self, model_path):
        model = actual_cnn.CNN().to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            frame_to_show, features = process_video_frame(frame)

            if len(features) > 0:
                letter = self.recognize_letter(features)
                if letter:
                    self.update_recognized_letter(letter)

            frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)

            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)

            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

    def recognize_letter(self, features):
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(feature_tensor)
            _, predicted_class = torch.max(output, 1)
            letter = chr(predicted_class.item() + ord('A'))

        return letter if predicted_class.item() < 26 else None

    def update_recognized_letter(self, letter):

        self.recognized_text += letter
        self.text_input.setText(self.recognized_text)

        letter_image = self.get_letter_image(letter)
        if letter_image:
            self.letter_label.setPixmap(QPixmap.fromImage(letter_image))

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
