<a href="https://drive.google.com/file/d/114hiTtY-MXnYKhMK7B-DZtJ9PebvCw7r/view?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

<p align="center">
<img src="https://user-images.githubusercontent.com/76246927/230722882-4395259b-ab3b-4f11-9f34-edd74b0b521d.png" width="600" height="200">
</p>


# YOLOv4 Custom Object Detection

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
1. **YOLOV4_Custom/images**
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
### Unzip Files

* Where you wish to extract the zip file from Google Drive as shown by the path in the cell below.


In our case text files should be saved in **YOLOV4_Custom/images** directory. For e.g. **image1.jpg** should have a text file **image1.txt**.

```Python
%cd {HOME}
!unzip "dataset.zip"
```

# **Step 5**
## *Creating Custom.names file*
Labels of our objects should be saved in **YOLOV4_Custom/custom.names** file, each line in the file corresponds to an object. In our case since we have two object class, the file should contain the following.

```
with_mask
without_mask
```
# **Step 5**
## *Creating Train and Test files*
The annotated photos can then be randomly split into train and test sets in a **90:10** ratio.

**YOLOV4_Custom/train.txt** The location of the train dataset should be listed in each file row.

**YOLOV4_Custom/test.txt** The location of the test dataset should be listed in each file row.



```
YOLOV4_Custom/images/0211d7dcb0aa6d66.jpg
YOLOV4_Custom/images/02e3c15f755cf2f9.jpg
YOLOV4_Custom/images/03b72249aed1fef0.jpg
YOLOV4_Custom/images/0561cd46d01b21ff.jpg
YOLOV4_Custom/images/05fa9bdfbd9204ab.jpg
```

To divide all image files into 2 parts. 90% for train and 10% for test, Upload the *`process.py`* in *`YOLOV4_Custom`* directory

This *`process.py`* script creates the files *`train.txt`* & *`test.txt`* where the *`train.txt`* file has paths to 90% of the images and *`test.txt`* has paths to 10% of the images.

You can download the process.py script from my GitHub.

**Run *`process.py`***
```Python
%cd {HOME }
# run process.py ( this creates the train.txt and test.txt files in our darknet/data folder )
!python process.py

```
```Python
# list the contents of data folder to check if the train.txt and test.txt files have been created 
!ls
```


# **Step 6**
## *Creating YOLO data file*
Make a file called `detector.data` in the `YOLOV4_Custom` directory that contains details about the train and test data sets.

```
classes = 2
train = YOLOV4_Custom/train.txt
valid = YOLOV4_Custom/test.txt
names = YOLOV4_Custom/obj.names
backup = YOLOV4_Custom/backup
```

# **Step 7**
## *Cloning Directory to use Darknet*
Darknet, an open source neural network framework, will be used to train the detector. Download and create a dark network

```Python
%cd {HOME}
!git clone https://github.com/AlexeyAB/darknet
```

# **Step 8** 
## *Make changes in the `makefile` to enable OPENCV and GPU*

# change makefile to have GPU and OPENCV enabled
# also set CUDNN, CUDNN_HALF and LIBSO to 1
```Python
%cd {HOME}/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile
```

## *Run `make` command to build darknet*
```Python
# build darknet 
!make
```

# **Step 9**
## *Making changes in the yolo Configuration file*

Download the `yolov4-custom.cfg` file from `darknet/cfg` directory, make changes to it, and upload it to the `YOLOV4_Custom` folder on your drive .


**Make the following changes:**

1. `batch=64`  (at line 6)
2. `subdivisions=16`  (at line 7)

3. `width = 416` (has to be multiple of 32, increase height and width will increase accuracy but training speed will slow down).  (at line 8)
4. `height = 416` (has to be multiple of 32).  (at line 9)

5. `max_batches = 10000` (num_classes*2000 but if classes are less then or equal to 3 put `max_batches = 6000`)  (at line 20)

6. `steps = 8000, 9000` (80% of max_batches), (90% of max_batches) (at line 22)
 
7. `classes = 2` (Number of your classes) (at line 970, 1058, 1146)
8. `filters = 21` ( (num_classes + 5) * 3 )  (at line 963, 1051, 1139)

Save the file after making all these changes, and upload it to the `YOLOV4_Custom` folder on your drive .


# **Step 10**
## *Downloading Pre-trained weights*
To train our object detector, we can use the pre-trained weights that have already been trained on a large data sets.
```Python
# changing the current drive to the pre-trained-weights directory to download pretrained weights 
%cd {HOME}/pre-trained-weights

# Download the yolov4 pre-trained weights file
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

# **Step 11**
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

#  **Step 12**
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



