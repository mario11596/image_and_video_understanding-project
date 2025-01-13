
# A transcription tool: from sign language to text

Hand sign detection is a widely researched field, that has resulted in many published approaches.
Machine learning algorithms form the core of most of the recently published methods.
Our project is build upon the state-of-the-art in sign language detection, and that allows users to write simple texts using sign language though a user interface. Signs performed by the user are captured by the webcam and classified in real-time by our pre-trained model. The classified signs will then get translated into alphabetic characters and displayed on the screen. Besides the standard American alphabet, we introduce seven additional signs: comma, dot, question mark, exclamation mark, minus, delete and space.

## Python packages
At the beginning you should install all requires Python packages using Python installer pip with following command:
```bash
pip install -r requirements.txt
```
We suggest to you python version 3.11 since that the project is done in that version.

## Dataset
The whole dataset is created by us. If you want to create your own dataset, you should run the script collect_dataset.py and define how many images you want per class. Run the script dataset_preparing.py. It will split the dataset on the three parts training, validation and test. It will be created the new folder dataset which is used in the other scripts.

## MediaPipe
Run the script featureExtractionImage.py in folder featureDetection to create hand landmarks for all three datasets. It will be created three .npy files.

## CNN Model

## Residual Neural Network Model
To run RNN model, you have to run the script rnn_model_training_testing.py. To do training, you have to setup training_validation_mode variable on True, while for testing you need to setup up on False. It will automatically load training and validation dataset using .npy files. If you do not want to do training, you can use our last pretrained model sign_language_rnn_model.pth which is available in this repository. The model has a accuracy of 99.19%

## Interface
There are two types of interfaces. One is for Windows and Linux (interface_Windows.py), while the second is for macOS (interface_macOS.py).  The interface consists of three parts. The first is the live video streaming. The second is the image of the sign which model predictes, and the last one is the text box where each sign is showed in the form of text. Moreover, we add the button, so that user can delete current text from the text box. To stop using the interface, just close the window or stop the program.