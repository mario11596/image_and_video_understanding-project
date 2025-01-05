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
model_path = "sign_language_rnn_model.pth"
class_mapping = [
                'A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'minus', 'N', 'O', 'P', 'period', 'Q', 'question mark', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z'
                ]
text_box_size = 50
button_box_size = 40
timmer_camera_frame = 30
timmer_new_prediction = 2000


# Interface of the application with setting of the layouts
class SignLanguageRecognition(QWidget):
    def __init__(self):
        super().__init__()

        self.main_layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()

        self.splitter = QSplitter(Qt.Horizontal)

        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.camera_label)

        self.letter_label = QLabel("Recognized Letter")
        self.letter_label.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.letter_label)

        self.splitter.setSizes([740, 540])  # Set initial sizes for splitter panes

        self.top_layout.addWidget(self.splitter)
        self.main_layout.addLayout(self.top_layout)

        self.text_input = QLineEdit()
        self.text_input.setReadOnly(True)

        font = QFont()
        font.setPointSize(18)
        self.text_input.setFont(font)
        self.text_input.setFixedHeight(text_box_size)

        self.main_layout.addWidget(self.text_input)

        self.clear_button = QPushButton("Clear input text")
        self.clear_button.setFixedHeight(button_box_size)
        self.clear_button.setStyleSheet("font-size: 18px;")
        self.clear_button.clicked.connect(self.clear_textbox)
        self.main_layout.addWidget(self.clear_button)

        self.setLayout(self.main_layout)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Camera not found or is unavailable.")
            return

        self.real_time_update = QTimer(self)
        self.real_time_update.timeout.connect(self.refresh_camera_frame)
        self.real_time_update.start(timmer_camera_frame)

        self.process_timer = QTimer(self)
        self.process_timer.timeout.connect(self.process_frame)
        self.process_timer.start(timmer_new_prediction)

        self.recognized_output_text = ""
        self.latest_letter = None

        self.model = self.load_model()
        self.latest_features = None

    # Load the trained RNN model
    def load_model(self):
        model = rnn_model.ResidualNeuralNetworkModel().to(device)

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    # Clear text from the text box
    def clear_textbox(self):
        self.recognized_output_text = ""
        self.text_input.clear()

    # Show on camera new frame with landmarks
    def refresh_camera_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_to_show, features = process_video_frame(frame)
            self.latest_features = features

            transfrom_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            h, w, ch = transfrom_rgb.shape
            qimg = QImage(transfrom_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

    def process_frame(self):
        if len(self.latest_features) > 0:

            sign = self.recognize_sign(self.latest_features)
            if sign:
                self.latest_letter = sign
                self.update_recognized_sign(sign)
        else:
            self.update_recognized_sign(None)


    # Predict the sign using the model
    def recognize_sign(self, features):
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(feature_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted_class = torch.max(output, 1)
            class_index = predicted_class.item()
            certainty = round(probabilities[0, class_index].item() * 100, 2)

            print("Certainty: ", certainty, " % certainty")

            # Check if the index is valid in the class mapping
            if 0 <= class_index < len(class_mapping):
                letter = class_mapping[class_index]
            else:
                letter_image = self.set_sign_image(None)
                self.letter_label.setPixmap(QPixmap.fromImage(letter_image))
        return letter

    # Show the sign on the right part of application window and put the sign in the textbox
    def update_recognized_sign(self, sign):
        if sign == "del":
            self.recognized_output_text = self.recognized_output_text[:-1]
        elif sign == "Space":
            self.recognized_output_text += " "
        elif sign == "exclamation mark":
            self.recognized_output_text += "!"
        elif sign == "period":
            self.recognized_output_text += "."
        elif sign == "question mark":
            self.recognized_output_text += "?"
        elif sign == "comma":
            self.recognized_output_text += ","
        elif sign == "minus":
            self.recognized_output_text += "-"
        elif sign is not None:
            self.recognized_output_text += sign

        self.text_input.setText(self.recognized_output_text)

        letter_image = self.set_sign_image(sign)
        self.letter_label.setPixmap(QPixmap.fromImage(letter_image))

    # Get the sign image
    def set_sign_image(self, sign):
        if sign is not None:
            img = Image.open(f"letters/{sign}.png")
        else:
            img = Image.open(f"letters/NoSign.png")

        img = img.convert("RGB")
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)

        return qimg

app = QApplication([])
window = SignLanguageRecognition()
window.show()
app.exec()