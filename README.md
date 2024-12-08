
# A transcription tool: from sign language to text

Hand sign detection is a widely researched field, that has resulted in many published approaches.
Machine learning algorithms form the core of most of the recently published methods.
Our project is build upon the state-of-the-art in sign language detection, and that allows users to write simple texts using sign language though a user interface. Signs performed by the user are captured by the webcam and classified in real-time by our pre-trained model. The classified signs will then get translated into alphabetic characters and displayed on the screen. Besides the standard American alphabet, we introduce eight additional signs: comma, dot, parentheses, question mark, exclamation mark, minus, delete and space.

## Python Packages
At the beginning you should install all requires Python packages using Python installer pip with following command:
```bash
pip install -r requirements.txt
```

## Dataset
Run the script dataset_preparing.py. It will download the dataset from Kaggle page and combine images from that dataset and custom dataset into one folder. After that step, it will be run the script to generate validation/training/testing dataset

## MediaPipe
Run the script featureExtractionImage.py in folder featureDetection to create hand landmarks for datasets

## CNN Model

## RNN Model
To run RNN model, you have to run the script rnn_model.py. It will automatically load training and validation dataset using .npy files. After the training is done, "training_validation_mode" variable in line 15 has to be changed into False, and that is possible to do the testing. The accuracy will be 74.68% 


