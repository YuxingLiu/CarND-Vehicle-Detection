# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repository presents an image processing pipeline to identify vehicles in a video from a front-facing camera on a car.

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

[//]: # (Image References)

[image1]: ./output_images/example_car_noncar.png "Camera Calibration"


---


## Linear SVM Classifier

### Dataset Summary

The code for this step is in the notebook [`Load and Save Data.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Load%20and%20Save%20Data.ipynb).  

For this project, a labeled dataset of 8792 [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip) and 8968 [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples are used to train the classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the [project video](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/project_video.mp4) itself. Here are several sample images in the vehicle and non-vehicle classes:

![alt text][image1]


### Color Histograms and Spatial Binned Color Features


### Histogram of Oriented Gradients (HOG) Features


### Linear SVC Training


## Hog Sub-sampling Window Search

## Video Implementation

Here's a link to [project video result](./test_videos_output/project_video.mp4).


## Discussion
