
# A transcription tool: from sign language to text

Hand sign detection is a widely researched field, that has resulted
in many published approaches. Machine learning algorithms form the core
of most of the recently published methods. Our project is build upon the
state-of-the-art in sign language detection, and that allows users to write
simple texts using sign language though a user interface. Signs performed
by the user are captured by the webcam and classified in real-time by our
pre-trained model. The classified signs will then get translated into
alphabetic characters and displayed on the screen. Besides the standard
American alphabet, we introduce seven additional signs: comma, dot,
question mark, exclamation mark, minus, delete and space.

## 1. Install needd Python packages
At the beginning you should install all requires Python packages using Python
installer pip with following command:
```bash
pip install -r requirements.txt
```
We suggest to you python version **3.11** since that the project is done in that
version.

## 2. Dataset
There are two options available:

**a) Create your own dataset:**
The whole dataset is created was created us. If you want to try it yourself,
 - run the script **collect_dataset.py** and define how many images you want per
class.
 - run the script **dataset_preparing.py** to automatically split the dataset 
into the three parts training, validation and testing which is used for the
further steps.

**b) Download our dataset:**
Download link https://cloud.tugraz.at/index.php/s/kXiNPM86rZQY4xi. Please place 
the downloaded images in the **./dataset** folder.

## 3. Features Extraction with MediaPipe
**a) Do feature extraction for image dataset:**
- run **./featureDetection/featureExtractionImage.py**.
Three .npy files will be created for training, testing and validating the model.

If you want to know more about the extracted Features,
take a look at the README.md in the ./featureDetection folder.

## 4. Residual Neural Network Model
**a) Train model:**
- set **traning_validation_mode** variable in script
**rnn_model_training_testing.py** to **True**
- run rnn_model_training_testing.py

**b) Test model:**
- set **traning_validation_mode** variable in script
**rnn_model_training_testing.py** to **False**
- run rnn_model_training_testing.py

**c) Use (Existing) model:**

After training you will receive your pretrained model
as a .pth file. 
To now use this model for the application, use 
Strg+Shift+F to search for *"Your pretrained model"* and
replace the .pth file with the model you want to use.

If you do not want to do training on your own, you can
use our latest pretrained model
**pretrained_rnn_model.pth** which you can find in the
folder. This model's stats are: accuracy 98.89%, precision 0.98954, recall 0.98899 and F1 score 0.98914.

## 5. Interface
To launch our application, there are two options:
**a) Windows & Linux interface:**
- run **interface_Windows.py**

**b) macOS interface:**
- run **interface_macOS.py**

The interface consists of three parts. The first is the
live video streaming. The second is the image of the sign
which model predicts, and the last one is the text box
where each sign is shown in the form of text. Moreover,
we add the button, so that user can delete current text
from the text box. Simply run the python script and place
your hand in front of the camera. To stop using the
interface, just close the window or stop the program.