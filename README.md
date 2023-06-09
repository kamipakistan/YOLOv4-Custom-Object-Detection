<p align="center">
<img src="https://user-images.githubusercontent.com/76246927/230722882-4395259b-ab3b-4f11-9f34-edd74b0b521d.png" width="600" height="300">
</p>


# YOLOv4 Custom Object Detection
YOLOv4 is a state-of-the-art real-time object detection algorithm that stands for "You Only Look Once version 4". It was developed by a group of researchers at the University of Washington and was released in 2020.

YOLOv4 builds upon the success of its predecessors, YOLOv3 and YOLOv2, and incorporates several advancements to improve the accuracy and speed of object detection. It is considered one of the best object detection models available, with the ability to detect objects in real-time video streams, making it suitable for a wide range of applications, including security, autonomous driving, and robotics.

YOLOv4 is a deep learning model that utilizes convolutional neural networks (CNNs) and a single-shot detection approach, meaning that it processes the entire image at once to detect objects instead of segmenting the image into smaller parts. This makes it much faster than other object detection algorithms.

Additionally, YOLOv4 uses a novel architecture that includes multiple backbones and prediction heads to improve the accuracy of object detection. It also employs advanced data augmentation techniques and incorporates techniques such as focal loss, spatial pyramid pooling, and self-adversarial training to enhance the model's performance.

Overall, YOLOv4 is a highly effective and efficient object detection algorithm that is widely used in computer vision applications. In this blog, we will explore how to use YOLOv4 for custom object detection, allowing you to train the model to recognize objects specific to your needs.

# **Step 1**
## Enabling and testing the GPU

The notebook's GPUs must first be enabled:

- Select Notebook Settings under Edit.
- choose GPU using the Hardware Accelerator drop-down

Next, we'll check if Tensorflow can connect to the GPU: By executing the following code, you may quickly determine whether the GPU is enabled.

```Python
# Check if NVIDIA GPU is enabled
!nvidia-smi
```
<img src = https://user-images.githubusercontent.com/76246927/230774198-81397660-b080-4f40-8778-937b7350e15f.png width = 500, height = 300>

```
# verify CUDA
!/usr/local/cuda/bin/nvcc --version
```

# **Step 2**
Mounting the Drive to store and load files.

```Python
from google.colab import drive
drive.mount('/content/gdrive')
```
Let's make the required files and directories one by one for training with custom objects.

1. **YOLOV4_Custom**
2. **YOLOV4_Custom/custom.names** 
3. **YOLOV4_Custom/train.txt**
4. **YOLOV4_Custom/test.txt**
5. **YOLOV4_Custom/backup**
6. **YOLOV4_Custom/detector.data**
7. **YOLOV4_Custom/cfg**
8. ****YOLOV4_Custom/cfg/yolov4-custom.cfg****

**Changing directory to drive Directory** 

```Python
# changing directory to the google drive
import os
drive_path = os.path.join(os.getcwd(), "gdrive/MyDrive")
%cd {drive_path}
```

# **Step 3**

Create a folder named ***YOLOV4_Custom*** in your drive. 
 
Next, create another folder named ***backup***
inside the ***YOLOV4_Custom*** folder. This is where we will save our trained weights (This path is mentioned in the ***obj.data*** file which we will upload later) 

```Python
HOME = os.path.join(drive_path, "YOLOV4_Custom")
HOME
```

```Python
# Making YOLOV4_Custom directory
os.mkdir(f"{HOME}")

# changing current directory to the HOME directory.
%cd {HOME}

# Making backup directory inside YOLOV4_Custom
os.mkdir("backup")
```

# **Step 4**
# Upload and Unzip Files
I am going to train a YOLOv4 model on the Plant Disease Detection dataset, which is publicly available for download from this [link](https://www.kaggle.com/datasets/kamipakistan/plant-diseases-detection-dataset). This dataset is a collection of images of plants that have been labeled as healthy or diseased with one of 30 different diseases. The dataset was created by a group of researchers at the Indian Institute of Technology and is intended for use in training machine learning models for plant disease detection. By training a YOLOv4 model on this dataset, I hope to create an accurate and efficient system for detecting plant diseases early, which can help improve crop yields and prevent significant losses for farmers.

*I first uploaded the dataset to my Google Drive and then unzipped it in the main 'YOLOV4_Custom' directory with the following `unzip`command.*

```Python
%cd {HOME}
!unzip "PlantDisease416x416.zip"
```

# **Step 5**
## *Creating Custom.names file*
To ensure that our YOLOv4 model can accurately identify the 30 different classes of objects in our dataset, we need to save the labels of these objects in a file called **`custom.names`**, which should be saved inside the **'YOLOV4_Custom'** directory. Each line in this file corresponds to one of the object classes in our dataset. In our case, since we have 30 different classes of plant diseases and healthy plants, the 'custom.names' file should contain one line for each of these 30 classes, so that our model can correctly recognize and classify them.

**custom.names**
```
Apple Scab Leaf
Apple leaf
Apple rust leaf
Bell_pepper leaf
Bell_pepper leaf spot
Blueberry leaf
Cherry leaf
Corn Gray leaf spot
Corn leaf blight
Corn rust leaf
Peach leaf
Potato leaf
Potato leaf early blight
Potato leaf late blight
Raspberry leaf
Soyabean leaf
Soybean leaf
Squash Powdery mildew leaf
Strawberry leaf
Tomato Early blight leaf
Tomato Septoria leaf spot
Tomato leaf
Tomato leaf bacterial spot
Tomato leaf late blight
Tomato leaf mosaic virus
Tomato leaf yellow virus
Tomato mold leaf
Tomato two spotted spider mites leaf
grape leaf
grape leaf black rot
```


# **Step 6**
## *Creating Train and Test files*
After uploading and unzipping the dataset, the annotated images should be split into train and test sets with a ratio of **80:20**. The location of the images in the train and test sets should be listed in separate files: **YOLOV4_Custom/train.txt** and **YOLOV4_Custom/test.txt**. Each file row should contain the location of one image in the respective dataset. These files will be used during training to access the images in the correct location.

```
../PlantDisease416x416/train/Image416_1912.jpg
../PlantDisease416x416/train/testImage416_215.jpg
../PlantDisease416x416/train/Image416_673.jpg
../PlantDisease416x416/train/Image416_1635.jpg
../PlantDisease416x416/train/Image416_1824.jpg
../PlantDisease416x416/train/Image416_1023.jpg
../PlantDisease416x416/train/Image416_23.jpg
../PlantDisease416x416/train/Image416_2226.jpg
```

To divide all image files into 2 parts. 80% for train and 20% for test, Upload the *`process.py`* in *`YOLOV4_Custom`* directory

This *`process.py`* script creates the files *`train.txt`* & *`test.txt`* where the *`train.txt`* file has paths to 80% of the images and *`test.txt`* has paths to 20% of the images.

You can download the process.py script from my GitHub.

**Open `process.py` specify the path and then run it.**
```Python
%cd {HOME }
# run process.py ( this creates the train.txt and test.txt files in our darknet/data folder )
!python process.py

```
```Python
# list the contents of data folder to check if the train.txt and test.txt files have been created 
!ls
```


# **Step 7**
## *Creating Configuration file for YOLOv4 model training*
Make a file called `detector.data` in the `YOLOV4_Custom` directory.

```

classes = 30
train = ../train.txt
valid = ../test.txt
names = ../custom.names
backup = ../backup
```

* The classes variable indicates the total number of object classes (in this case, 30).
* train and valid variables point to the text files containing the file paths for the training and validation sets, respectively.
* The names variable points to the file containing the names of the object classes, with one class per line.
* Finally, backup points to the directory where the weights of the model will be saved during training.



# **Step 8**
## *Cloning Directory to use Darknet*
Darknet, an open source neural network framework, will be used to train the detector. Download and create a dark network

```Python
%cd {HOME}
!git clone https://github.com/AlexeyAB/darknet
```

# **Step 9** 
## *Make changes in the `makefile` to enable OPENCV and GPU*

```Python
# change makefile to have GPU and OPENCV enabled
# also set CUDNN, CUDNN_HALF and LIBSO to 1

%cd {HOME}/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile
```
* The `%cd {HOME}/darknet/` command changes the current directory to the darknet directory in the HOME directory.

* The first command replaces the string OPENCV=0 with OPENCV=1 in the Makefile. This is done to enable OpenCV support in Darknet, which is necessary for some image-related tasks.
* The second command replaces the string GPU=0 with GPU=1 in the Makefile. This is done to enable GPU acceleration in Darknet, which can greatly speed up training and inference.
* The third command replaces the string CUDNN=0 with CUDNN=1 in the Makefile. This is done to enable cuDNN support in Darknet, which is an NVIDIA library that provides faster implementations of neural network operations.
* The fourth command replaces the string CUDNN_HALF=0 with CUDNN_HALF=1 in the Makefile. This is done to enable mixed-precision training in Darknet, which can further speed up training and reduce memory usage.

## *Run `make` command to build darknet*
The `!make` command is a Linux command-line instruction that invokes the make utility to compile and build the Darknet codebase based on the configurations specified in the Makefile. This command reads the Makefile in the current directory and compiles the source code by executing various build commands specified in the Makefile. After the compilation process is complete, the make utility generates an executable binary file that can be used to run various Darknet commands and utilities.

```Python
# build darknet 
!make
```

# **Step 10**
## *Making changes in the yolo Configuration file*

Download the `yolov4-custom.cfg` file from `darknet/cfg` directory, make changes to it, and upload it to the `YOLOV4_Custom` folder on your drive .


**Make the following changes:**

1. `batch=64`  (at line 6)
2. `subdivisions=16`  (at line 7)

3. `width = 416` (has to be multiple of 32, increase height and width will increase accuracy but training speed will slow down).  (at line 8)
4. `height = 416` (has to be multiple of 32).  (at line 9)

5. `max_batches = 60000` (num_classes*2000 but if classes are less then or equal to 3 put `max_batches = 6000`)  (at line 20)

6. `steps = 48000, 54000` (80% of max_batches), (90% of max_batches) (at line 22)
 
7. `classes = 30` (Number of your classes) (at line 970, 1058, 1146)
8. `filters = 105` ( (num_classes + 5) * 3 )  (at line 963, 1051, 1139)

Save the file after making all these changes, and upload it to the `YOLOV4_Custom` folder on your drive .



# **Step 11**
## *Downloading Pre-trained weights*
To train our object detector, we can use the pre-trained weights that have already been trained on a large data sets.
```Python
# changing the current drive to the pre-trained-weights directory to download pretrained weights 
%cd {HOME}/pre-trained-weights

# Download the yolov4 pre-trained weights file
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

# **Step 12**
## *Training the model*
As soon as we have all the necessary files and annotated photographs, we can begin our training.
Up till the loss reaches a predefined level, we can keep training. Weights for the custom detector are initially saved once every 100 iterations until 1,000 iterations, after which they are saved once every 10,000 iterations by default.

We can do detection using the generated weights after the training is finished.

```Python
%cd {HOME}/darknet
!./darknet detector train {HOME}/obj.data {HOME}/yolov4-custom.cfg {HOME}/pre-trained-weights/yolov4.conv.137.1 -dont_show -map
```

## *Continue training from where you left*
Continue training from where you left off, your Model training can be stopped due to multiple reasons, like the notebook time out, notebook craches, due to network issues,  and many more,  so you can start your training from where you left off, by passing the previous trained weights. The weights are saved every 100 iterations as ***yolov4-custom_last.weights*** in the **YOLOV4_Custom/backup** folder on your drive.

```Python
# To start training your custom detector from where you left off(using the weights that were last saved)

%cd {HOME}/darknet
!./darknet detector train {HOME}/obj.data {HOME}/yolov4-custom.cfg {HOME}/backup/yolov4-custom_last.weights  -dont_show -map
```

#  **Step 13**
## *Check performance* 

```Python
# define helper function imShow
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  
```
**Check the training chart**

```Python
#only works if the training does not get interrupted
%cd {HOME}/darknet
imShow('chart.png')
```

<img src="https://user-images.githubusercontent.com/76246927/230772096-1dcd8dc6-d092-4b1f-8b78-baeaac43f51a.png" width="500" height="500">

**Check mAP (mean average precision)**
```Python
#You can check the mAP for all the saved weights to see which gives the best results 
%cd {HOME}/darknet
!./darknet detector map {HOME}/obj.data {HOME}/yolov4-custom.cfg {HOME}/backup/yolov4-custom_best.weights -points 0
```

# **Step 14** 
## *Test your custom Object Detector*

**Make changes to your custom config file**
*   change line batch to batch=1
*   change line subdivisions to subdivisions=1

You can do it either manually or by simply running the code below
```Python
#set your custom cfg to test mode 
%cd {HOME}
!sed -i 's/batch=64/batch=1/' yolov4-custom.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-custom.cfg
```

## *Run detector on an image*

```Python
# run your custom detector with this command (upload an image to your google drive to test, the thresh flag sets the minimum accuracy required for object detection)
%cd {HOME}/darknet
!./darknet detector test {HOME}/obj.data {HOME}/yolov4-custom.cfg {HOME}/backup/yolov4-custom_best.weights {HOME}/player.jpg -thresh 0.5 
```

```Python
imShow('predictions.jpg')
```


To perform detection realtime see inferece files directory
