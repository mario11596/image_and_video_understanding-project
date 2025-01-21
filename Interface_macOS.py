#Known Issues - Warnings
# - Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
#   https://github.com/google-ai-edge/mediapipe/issues/5639 (known issue from Mediapipe)
# - +[IMKClient subclass]: chose IMkClient_Modern, +[IMKInputSession subclass]: chose IMKInputSession_Modern
#   https://discussions.apple.com/thread/255761734?sortBy=rank
# - Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
#   https://github.com/google-ai-edge/mediapipe/issues/5462
#   Solution: downgrade mediapipe to version 0.10.9 with pip install mediapipe==0.10.9
# - WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
#  https://discuss.ai.google.dev/t/warning-all-log-messages-before-absl-initializelog-is-called-are-written-to-stderr-e0000-001731955515-629532-17124-init-cc-229-grpc-wait-for-shutdown-with-timeout-timed-out/50020
#  Solution: downgrade to grpcio==1.67.1

import customtkinter
import cv2
from PIL import Image, ImageTk
import threading
import os
import subprocess
from tkinter import PhotoImage
import torch

import rnn_model
import time
from featureDetection.live_detection import process_video_frame
os.environ["PYTHONWARNINGS"] = "ignore"

#set up model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "sign_language_rnn_model.pth"
model_path_old = "sign_language_rnn_model_old.pth"
class_mapping = ['A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation\nmark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'minus', 'N', 'O', 'P', 'period', 'Q', 'question\nmark', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X',
                'Y', 'Z']

#every 0.5 seconds model should give new prediction
update_sign_detection_sec = 0.5 #can be adapted but we found this to be a reasonable number
#retrieve pretrained model
model = rnn_model.ResidualNeuralNetworkModel().to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

last_sign = ""
last_percent = ""

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.attributes("-fullscreen", True)
root.title("Webcam Viewer and Label Display")

#store width of window for resizing
last_width = root.winfo_width()

#used for testing to switch between pre-trained models to compare the results
'''def switch_model():
    global model
    global state_dict
    if change_model_button.cget("text") == "Use old Model":
        model = rnn_model_old.ResidualNeuralNetworkModel().to(device)
        state_dict = torch.load(model_path_old, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        subtitle_label.configure(text="You are using the old model")
        change_model_button.configure(text="Use new Model")
        customtkinter.set_appearance_mode("light")
        customtkinter.set_default_color_theme("blue")
    else:
        model = rnn_model.ResidualNeuralNetworkModel().to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        subtitle_label.configure(text="You are using the new model")
        change_model_button.configure(text="Use old Model")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
'''

def open_image_window():
    img1_path = "letters/SignsAlphabeth.png"
    img2_path = "letters/SignsSpecial.png"
    #open a new window to show possible characters
    if os.name == 'nt':  #windows
        os.startfile(img1_path)
        os.startfile(img2_path)
    elif os.name == 'posix':  #for macOS
        subprocess.run(["open", img1_path])
        subprocess.run(["open", img2_path])

def clear_text():
    text_history.configure(text="")

def update_font_size(event=None):
    global last_width

    #current width of the window
    current_width = root.winfo_width()

    #adapt font size depending on size of window
    if current_width != last_width:
        percent = 0.04
        if len(label_below1.cget("text")) > 1:
            percent = 0.02

        font_size = int(current_width * percent)
        font_size_percent = int(current_width * 0.02)

        label_below1.configure(font=("Helvetica", font_size))
        label_below2.configure(font=("Helvetica", font_size_percent))

        last_width = current_width

def set_history_text(sign, text):
    #update text box with latest recognized signs (and delete if necessary)
    if len(text) >= 50:
        text = text[1:]

    if sign == "comma":
        return text + ","
    elif sign == "del":
        return text[:-1]
    elif sign == "exclamation\nmark":
        return text + "!"
    elif sign == "minus":
        return text + "-"
    elif sign == "period":
        return text + "."
    elif sign == "question\nmark":
        return text + "?"
    elif sign == "space":
        return text + " "
    elif text == "":
        return sign
    else:
        return text + sign

def update_webcam():
    #open up webcam or update if already open
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        webcam_label.configure(text="Unable to access webcam.")
        return

    def recognize_sign(features):
        #give features to model and retrieve character from extracted features
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(feature_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted_class = torch.max(output, 1)
            class_index = predicted_class.item()
            certainty = round(probabilities[0, class_index].item() * 100, 0)

            #check if index is valid in the class mapping
            if 0 <= class_index < len(class_mapping):
                letter = class_mapping[class_index]
            else:
                letter = ""
        return letter, certainty

    def capture():
        global last_sign
        global last_percent
        last_recognition_time = time.time()  #init last recognition time
        while True:
            ret, frame = cap.read()
            #capture frame and proces features
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape
                aspect_ratio = w / h

                #resize frame to fit label while maintaining aspect ratio
                current_width = int(root.winfo_width() * 0.8)
                target_height = int(current_width / aspect_ratio)

                #get features out of captured frame
                frame = cv2.flip(frame, 1)
                frame_to_show, features = process_video_frame(frame)

                frame_resized = cv2.resize(frame_to_show, (current_width, target_height))
                img = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                webcam_label.configure(image=img)
                webcam_label.image = img

                #update Recognize Signs every update_sign_detection_sec seconds
                if len(features) > 0 and time.time() - last_recognition_time >= update_sign_detection_sec:
                    sign, percent = recognize_sign(features)
                    if sign:
                        #update labels
                        label_below1.configure(text=last_sign)
                        last_sign = sign
                        label_below2.configure(text=str(last_percent) + " %")
                        last_percent = percent
                        text_history.configure(text=set_history_text(last_sign, text_history.cget("text")))

                    last_recognition_time = time.time()  #update last recognition time
                elif len(features) == 0: #nothing was detected
                    label_below1.configure(text=last_sign)
                    last_sign = "Nothing\ndetected"
                    label_below2.configure(text=str(last_percent) + " %")
                    last_percent = "_"

            else:
                break

    threading.Thread(target=capture, daemon=True).start()

top_frame = customtkinter.CTkFrame(master=root)
top_frame.pack(pady=10, padx=10, fill="x")

title_label = customtkinter.CTkLabel(master=top_frame, text="🖖🤞✌️ Sign Recognizer", font=("Helvetica", 40))
title_label.pack(side="left", pady=10, padx=20)

#Used for testing
#subtitle_label = customtkinter.CTkLabel(master=top_frame, text="You are using the new model", font=("Helvetica", 15))
#subtitle_label.pack(side="left", pady=10, padx=20)

open_images_button = customtkinter.CTkButton(master=top_frame, text="Show possible Signs", command=open_image_window, fg_color="#738678", hover_color="#505e54")
open_images_button.pack(side="right", padx=20)

#Used for testing
#change_model_button = customtkinter.CTkButton(master=top_frame, text="Use old Model", command=switch_model, fg_color="#738678", hover_color="#505e54")
#change_model_button.pack(side="right", padx=20)

middle_frame = customtkinter.CTkFrame(master=root)
middle_frame.pack(pady=10, padx=10, fill="both", expand=True)

webcam_label = customtkinter.CTkLabel(master=middle_frame, text="", anchor="center")
webcam_label.pack(side="left", padx=20, pady=20, fill="y", expand=True)

labels_frame = customtkinter.CTkFrame(master=middle_frame, width=200)
labels_frame.pack(side="left", padx=20, pady=20, fill="y")

label_below1 = customtkinter.CTkLabel(master=labels_frame, text="Label 1", anchor="center", width=180, font=("Helvetica", 20))
label_below1.pack(pady=10, padx=10, fill="x", expand=True)

label_below2 = customtkinter.CTkLabel(master=labels_frame, text="Label 2", anchor="center", width=180, font=("Helvetica", 20))
label_below2.pack(pady=10, padx=10, fill="x", expand=True)

root.bind("<Configure>", update_font_size)

bottom_frame = customtkinter.CTkFrame(master=root)
bottom_frame.pack(pady=10, padx=10, fill="x")

text_history = customtkinter.CTkLabel(master=bottom_frame, text="", width=180, font=("Helvetica", 20))
text_history.pack(side="left", pady=10, padx=10, fill="x", expand=True)

clear_button = customtkinter.CTkButton(master=bottom_frame, text="Clear Text", command=clear_text, fg_color="#738678", hover_color="#505e54")
clear_button.pack(side="right", padx=20)

#start webcam
update_webcam()

icon_image = PhotoImage(file="letters/icon.png")
root.iconphoto(True, icon_image)

root.mainloop()
