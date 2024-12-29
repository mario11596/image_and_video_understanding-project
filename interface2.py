import tkinter as tk
from tkinter import Canvas, Label, StringVar
from PIL import Image, ImageTk
import cv2
import torch
from featureDetection.live_detection import process_video_frame
import rnn_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language")

        # Create main layout frames
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(fill=tk.X)

        # Left canvas for camera feed (1054x712)
        self.camera_canvas = Canvas(self.top_frame, bg="black", width=1054, height=712)
        self.camera_canvas.pack(side=tk.LEFT)

        # Right label for recognized letter (250x712)
        self.letter_label = Label(
            self.top_frame,
            text="Recognized Letter",
            bg="white",
            font=("Arial", 24),
            width=300,
            height=712,
            anchor="center"
        )
        self.letter_label.pack(side=tk.RIGHT, fill=tk.Y)

        # Text input for recognized text
        self.recognized_text_var = StringVar()
        self.text_input = Label(self.bottom_frame, textvariable=self.recognized_text_var, bg="white", font=("Arial", 14))
        self.text_input.pack(fill=tk.X, padx=10, pady=10)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Camera not found or is unavailable.")
            return

        # Load model
        self.model = self.load_model("sign_language_rnn_model.pth")

        self.recognized_text = ""

        # Start the update loop
        self.update_camera()

    def load_model(self, model_path):
        model = rnn_model.RNNModel().to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame

            # Resize the frame to fit into 1054x712 while maintaining aspect ratio
            target_width, target_height = 1054, 712
            original_height, original_width = frame.shape[:2]
            scale = min(target_width / original_width, target_height / original_height)
            resized_width = int(original_width * scale)
            resized_height = int(original_height * scale)

            # Resize the frame
            resized_frame = cv2.resize(frame, (resized_width, resized_height))

            # Create a black canvas of the target size
            canvas = cv2.cvtColor(cv2.copyMakeBorder(
                resized_frame,
                (target_height - resized_height) // 2,
                (target_height - resized_height + 1) // 2,
                (target_width - resized_width) // 2,
                (target_width - resized_width + 1) // 2,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            ), cv2.COLOR_BGR2RGB)

            # Process the frame for feature detection
            frame_to_show, features = process_video_frame(canvas)

            # Recognize letters if features are detected
            if len(features) > 0:
                letter = self.recognize_letter(features)
                if letter:
                    self.update_recognized_letter(letter)

            # Convert the processed frame to an ImageTk format
            img = Image.fromarray(canvas)
            imgtk = ImageTk.PhotoImage(image=img)

            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.camera_canvas.imgtk = imgtk

        self.root.after(30, self.update_camera)

    def recognize_letter(self, features):
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(feature_tensor)
            _, predicted_class = torch.max(output, 1)

            if 0 <= predicted_class.item() < 26:
                letter = chr(predicted_class.item() + ord('A'))
                #print(letter)
            else:
                letter = None
        return letter

    def update_recognized_letter(self, letter):
        self.recognized_text = letter
        self.recognized_text_var.set(self.recognized_text)

        letter_image = self.get_letter_image(letter)
        if letter_image:
            imgtk = ImageTk.PhotoImage(letter_image)
            self.letter_label.configure(image=imgtk)
            self.letter_label.imgtk = imgtk

    def get_letter_image(self, letter):
        try:
            letter_img_path = f"letters/{letter}.png"
            img = Image.open(letter_img_path).convert("RGB")
            return img
        except Exception as e:
            print(f"Error loading letter image: {e}")
            return None

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()