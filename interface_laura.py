from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QWidget, QSplitter, QPushButton
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import cv2
import torch
from PIL import Image
from featureDetection.live_detection import process_video_frame
import rnn_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_mapping = [
                'A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'minus', 'N', 'O', 'P', 'period', 'Q', 'question mark', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z'
                ]


# Interface of the application with setting of the layouts
class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.clear_text = None
        self.setWindowTitle("Sign Language Recognition")
        self.setFixedSize(1280, 720)  # Fixed window size

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        self.main_layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()

        self.splitter = QSplitter(Qt.Horizontal)

        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setScaledContents(True)
        self.splitter.addWidget(self.camera_label)

        self.letter_label = QLabel("Recognized Letter")
        self.letter_label.setScaledContents(True)
        self.splitter.addWidget(self.letter_label)

        self.splitter.setSizes([740, 540])  # Set initial sizes for splitter panes

        self.top_layout.addWidget(self.splitter)
        self.delete_button = QPushButton("Clear Text")
        self.delete_button.clicked.connect(self.clear_text)
        self.top_layout.addWidget(self.delete_button, alignment=Qt.AlignRight)
        self.main_layout.addLayout(self.top_layout)

        self.text_input = QLineEdit()
        self.text_input.setReadOnly(True)

        font = QFont()
        font.setPointSize(16)  # Larger font size for readability
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
        model = rnn_model.ResidualNeuralNetworkModel().to(device)
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
        if len(self.latest_features) > 0:

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
        if letter == "del":
            self.recognized_text = self.recognized_text[:-1]
        elif letter == "Space":
            self.recognized_text += " "
        elif letter == "exclamation mark":
            self.recognized_text += "!"
        elif letter == "period":
            self.recognized_text += "."
        elif letter == "question mark":
            self.recognized_text += "?"
        elif letter == "comma":
            self.recognized_text += ","
        elif letter == "minus":
            self.recognized_text += "-"
        else:
            self.recognized_text += letter

        self.text_input.setText(self.recognized_text)

        letter_image = self.get_letter_image(letter)

        if letter_image:
            self.letter_label.setPixmap(QPixmap.fromImage(letter_image))

    # Get the sign image
    def get_letter_image(self, letter):
        letter_img_path = f"letters/{letter}.png"

        img = Image.open(letter_img_path)
        img = img.convert("RGB")
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)

        return qimg


    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = SignLanguageApp()
    window.show()
    app.exec()