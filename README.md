# GENDER AND AGE DETECTION USING DEEP LEARNING 

## A. PROJECT SUMMARY

**Project Title:** Gender and Age Detection

![family](https://user-images.githubusercontent.com/44885554/114744544-6329f180-9d80-11eb-874b-4e89cc841c5d.png)

**Team Members:** 
- MUHAMMAD AKMAL BIN MOHD SABRI
- WAN MUHAMMAD ISMAT BIN WAN AZMY
- MUHAMMAD AKMAL KHAIRI BIN ABDUL HALIM
- MUHAMMAD IMRAN BIN ISMAIL

**Objectives:**
- To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture using Deep Learning on the Adience dataset.

## ACKNOWLEDGEMENTS

Our group makes use of the following open source projects:

 - [data-flair.training](https://data-flair.training/blogs/python-project-gender-age-detection/)

## B. ABSTRACT 

In this Python Project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face. We will use the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we make this a classification problem instead of making it one of regression.

**The CNN Architecture:**
The convolutional neural network for this python project has 3 convolutional layers:

- Convolutional layer; 96 nodes, kernel size 7
- Convolutional layer; 256 nodes, kernel size 5
- Convolutional layer; 384 nodes, kernel size 3

It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

- Detect faces
- Classify into Male/Female
- Classify into one of the 8 age ranges
- Put the results on the image and display it

<p align="center">
 <img src="https://user-images.githubusercontent.com/73923156/121848986-63f9e800-cd1d-11eb-8867-ef3b0c3a61d7.JPG" width=40% height=40%>
</p>

<p align="center">
Figure 1: AI output of detecting the user's gender & age.
</p>

## C.  DATASET

For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it here. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we will use have been trained on this dataset.

**Prerequisites:**

You’ll need to install OpenCV (cv2) to be able to run this project. You can do this with pip-

```python
  pip install opencv-python
```

Other packages you’ll be needing are math and argparse, but those come as part of the standard Python library.

**Purposes:**

- Detect human's gender either ‘Male’ and ‘Female’ in images
- Detect human's age in images
- Detect human's gender & age in real-time video streams

## D.   PROJECT STRUCTURE

The following directory is our structure of our project:
- ├── src
- │ └── age_deploy.protxt
- │ └── age_net.caffeemodel
- │ └── gad.py
- │ └── gender_deploy.protxt
- │ └── age_net.caffeemodel
- │ └── opencv_face_detector.pbtxt
- │ └── opencv_face_detector_uint8.pb
- ├── Dataset
- │ └── data1.jpg
- │ └── data2.jpg
- │ └── data3.jfif
- │ └── data4.jpg
- │ └── data5.jpg
- │ └── data6.jpg
- │ └── data7.jpg
- ├── result
- │ └── 1.jpg
- │ └── 2.jpg
- │ └── 3.jpg
- │ └── 4.jpg
- │ └── 5.jpg
- │ └── 6.jpg
- │ └── 7.jpg
- 14 files

The dataset/ directory contains the data described in the “Gender and Age Detection” section. Seven image examples/ are provided so that you can test the static image gender and age detector.

* opencv_face_detector.pbtxt
* opencv_face_detector_uint8.pb
* age_deploy.prototxt
* age_net.caffemodel
* gender_deploy.prototxt
* gender_net.caffemodel
* a few pictures to try the project on

For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

1. We use the argparse library to create an argument parser so we can get the image argument from the command prompt. We make it parse the argument holding the path to the image to classify gender and age for.

2. For face, age, and gender, initialize protocol buffer and model.

3. Initialize the mean values for the model and the lists of age ranges and genders to classify from.

4. Now, use the readNet() method to load the networks. The first parameter holds trained weights and the second carries network configuration.

5. Let’s capture video stream in case you’d like to classify on a webcam’s stream. Set padding to 20.

6. Now until any key is pressed, we read the stream and store the content into the names hasFrame and frame. If it isn’t a video, it must wait, and so we call up waitKey() from cv2, then break.

7. Let’s make a call to the highlightFace() function with the faceNet and frame parameters, and what this returns, we will store in the names resultImg and faceBoxes. And if we got 0 faceBoxes, it means there was no face to detect.
Here, net is faceNet- this model is the DNN Face Detector and holds only about 2.7MB on disk.

Create a shallow copy of frame and get its height and width.
Create a blob from the shallow copy.
Set the input and make a forward pass to the network.
faceBoxes is an empty list now. for each value in 0 to 127, define the confidence (between 0 and 1). Wherever we find the confidence greater than the confidence threshold, which is 0.7, we get the x1, y1, x2, and y2 coordinates and append a list of those to faceBoxes.
Then, we put up rectangles on the image for each such list of coordinates and return two things: the shallow copy and the list of faceBoxes.

8. But if there are indeed faceBoxes, for each of those, we define the face, create a 4-dimensional blob from the image. In doing this, we scale it, resize it, and pass in the mean values.

9. We feed the input and give the network a forward pass to get the confidence of the two class. Whichever is higher, that is the gender of the person in the picture.

10. Then, we do the same thing for age.

11. We’ll add the gender and age texts to the resulting image and display it with imshow().

* In the next two sections, we will train our Age & Gender detector.

## E.   TRAINING THE AGE & GENDER DETECTION

Let’s try this gender and age classifier out on some of our own images now.

We’ll get to the command prompt, run our script with the image option and specify an image to classify:

**Example 1**

```
D:\DEGREE UTeM\DEG SEM 3\AI\ASSIGNMENT\Gender and Age Detection\src>python gad.py --image data/data1.jpg
Gender: Male
Age: 25-32 years
Gender: Male
Age: 38-43 years
Gender: Male
Age: 38-43 years
```

**Figure 2:** Example python command to execute & evaluate the image from the dataset.

**Output:**

<p align="center">
<img src="https://user-images.githubusercontent.com/73923156/121850023-bbe51e80-cd1e-11eb-8bee-34b506d0a791.JPG" width=50% height=50%>
</p>

**Figure 3:** Program output after evaluating the image provided into the command.

***

**Example 2**

```
D:\DEGREE UTeM\DEG SEM 3\AI\ASSIGNMENT\Gender and Age Detection\src>python gad.py --image data/data2.jpg
Gender: Male
Age: 4-6 years
Gender: Male
Age: 8-12 years
Gender: Male
Age: 4-6 years
```

**Figure 4:** Example python command to execute & evaluate the second image from the dataset.

**Output:**

<p align="center">
 <img src="https://user-images.githubusercontent.com/73923156/121850123-e1722800-cd1e-11eb-9059-bf6425e02daf.JPG" width=50% height=50%>
</p>

**Figure 5:** Program output after evaluating the second image provided into the command.

***

**Example 3**

```
D:\DEGREE UTeM\DEG SEM 3\AI\ASSIGNMENT\Gender and Age Detection\src>python gad.py --image data/data3.jfif
Gender: Female
Age: 8-12 years
Gender: Male
Age: 8-12 years
Gender: Female
Age: 4-6 years
```

**Figure 6:** Example python command to execute & evaluate the third image from the dataset.

**Output:**

<p align="center">
 <img src="https://user-images.githubusercontent.com/73923156/121850370-3f067480-cd1f-11eb-884b-9dbe093e7c8f.JPG" width=50% height=50%>
</p>

**Figure 7:** Program output after evaluating the third image provided into the command.

***

**Example 4**

```
D:\DEGREE UTeM\DEG SEM 3\AI\ASSIGNMENT\Gender and Age Detection\src>python gad.py --image data/data4.jpg
Gender: Male
Age: 25-32 years
```

**Figure 8:** Example python command to execute & evaluate the fourth image from the dataset.

**Output:**

<p align="center">
 <img src="https://user-images.githubusercontent.com/73923156/121850475-69583200-cd1f-11eb-8ecb-fabf550f581e.JPG" width=50% height=50%>
</p>

**Figure 9:** Program output after evaluating the fourth image provided into the command.

***

**Example 5**

```
D:\DEGREE UTeM\DEG SEM 3\AI\ASSIGNMENT\Gender and Age Detection\src>python gad.py --image data/data5.jpg
Gender: Male
Age: 25-32 years
```

**Figure 10:** Example python command to execute & evaluate the fifth image from the dataset.

**Output:**

<p align="center">
 <img src="https://user-images.githubusercontent.com/73923156/121850689-b63c0880-cd1f-11eb-94d1-73f00918e9b8.JPG" width=50% height=50%>
</p>

**Figure 11:** Program output after evaluating the fifth image provided into the command.

***

**Example 6**

```
D:\DEGREE UTeM\DEG SEM 3\AI\ASSIGNMENT\Gender and Age Detection\src>python gad.py --image data/data7.jpg
Gender: Male
Age: 15-20 years
```

**Figure 12:** Example python command to execute & evaluate the sixth image from the dataset.

**Output:**

<p align="center">
  <img src="https://user-images.githubusercontent.com/58213194/121853705-d5d53000-cd23-11eb-81e0-b3b73e8edc2d.jpeg" width=50% height=50%>
</p>

**Figure 13:** Program output after evaluating the sixth image provided into the command.

***

<p align="center">
     <img width="800" alt="opencv_age_detection_confusion_matrix" src="https://user-images.githubusercontent.com/73923156/114961386-7b426380-9e9b-11eb-9994-2531d74e8633.png">
</p>

<p align="center">
     Figure 14: Age estimation confusion matrix benchmark
</p>

* The age groups 0-2, 4-6, 8-13 and 25-32 are predicted with relatively high accuracy. ( see the diagonal elements )
* The output is heavily biased towards the age group 25-32 ( see the row belonging to the age group 25-32 ). This means that it is very easy for the network to get confused between the ages 15 to 43. So, even if the actual age is between 15-20 or 38-43, there is a high chance that the predicted age will be 25-32. This is also evident from the Results section.

Apart from this, we observed that the accuracy of the models improved if we use padding around the detected face. This may be due to the fact that the input while training were standard face images and not closely cropped faces that we get after face detection.

We also analysed the use of face alignment before making predictions and found that the predictions improved for some examples but at the same time, it became worse for some. It may be a good idea to use alignment if you are mostly working with non-frontal faces.

As you can see, one of the biggest issues with the age prediction model is that it’s heavily biased toward the age group 25-32. Looking at Figure 14, we can see that our model may predict the 25-32 age group when in fact the actual age belongs to a different age bracket.

**You can combat this bias by:**

- Gathering additional training data for the other age groups to help balance out the dataset

- Applying class weighting to handle class imbalance

- Being more aggressive with data augmentation

- Implementing additional regularization when training the model

Given these results, we are hopeful that our model will generalize well to images outside our training and testing set.


## F.  RESULT AND CONCLUSION

Detecting Age & Gender with OpenCV in real-time

You can then launch the gender and age detector in real-time video streams using the following command:
- $ python gad.py

<p align="center">
     <img width="800" alt="opencv_age_detection_confusion_matrix" src="https://user-images.githubusercontent.com/73923156/114885569-5ae1bd00-9e39-11eb-9b26-8338096ac69c.JPG">
</p>

**Figure 15:** Age & Gender Detection in real-time video streams

In Figure 15, you can see that our Age & Gender detector is capable of running in real-time (and is correct in its predictions as well.

Overall, we think the accuracy of the models is decent but can be improved further by using more data, data augmentation and better network architectures. Thanks to our AI lecturer, Prof. Goh Ong Sing for giving us the opportunity to learn how to implement a real-world AI project using Python. We gain a lot of knowledge throughout this journey, AI is really interesting and crucial in our life.

## G.   PROJECT PRESENTATION 

In this python project, we implemented Convolutional Neural Network (CNN) to detect gender and age from a single picture of a face

**What is Computer Vision?**

Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. The challenges it faces largely follow from the limited understanding of biological vision. Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.

**What is OpenCV?**

OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

**What is a CNN?**

A Convolutional Neural Network is a deep neural network (DNN) widely used for the purposes of image recognition and processing and NLP. Also known as a ConvNet, a CNN has input and output layers, and multiple hidden layers, many of which are convolutional. In a way, CNNs are regularized multilayer perceptrons.

[![Watch the video](https://user-images.githubusercontent.com/44885554/121906192-7431b780-cd5d-11eb-80d5-30b6c6bf861f.png)](https://www.youtube.com/watch?v=x-UmU5zGxXk&ab_channel=AkmalKhairi)
