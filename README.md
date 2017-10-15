# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repository presents an image processing pipeline to identify vehicles in a video from a front-facing camera on a car.

![](./output_images/project.gif)

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

[image1]: ./output_images/example_car_noncar.png "Data Visualization"
[image2]: ./output_images/color_hist_car1.png "Color Histograms Car 1"
[image3]: ./output_images/color_hist_car2.png "Color Histograms Car 2"
[image4]: ./output_images/color_hist_noncar1.png "Color Histograms Not-car 1"
[image5]: ./output_images/color_hist_noncar2.png "Color Histograms Not-car 2"
[image6]: ./output_images/color_bins_car1.png "Color Binning Car 1"
[image7]: ./output_images/color_bins_car2.png "Color Binning Car 2"
[image8]: ./output_images/color_bins_noncar1.png "Color Binning Not-car 1"
[image9]: ./output_images/color_bins_noncar2.png "Color Binning Not-car 2"
[image10]: ./output_images/hog_car1.png "HOG Car 1"
[image11]: ./output_images/hog_car2.png "HOG Car 2"
[image12]: ./output_images/hog_noncar1.png "HOG Not-car 1"
[image13]: ./output_images/hog_noncar2.png "HOG Not-car 2"
[image14]: ./output_images/search_heat1.png "Window Search 1"
[image15]: ./output_images/search_heat2.png "Window Search 2"

---


## Linear SVM Classifier

### Dataset Summary

The code for this step is in the notebook [`Load and Save Data.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Load%20and%20Save%20Data.ipynb).  

For this project, a labeled dataset of 8792 [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip) and 8968 [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples were used to train the classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the [project video](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/project_video.mp4) itself. Here are several sample images in the vehicle and non-vehicle classes:

![alt text][image1]


### Color Histograms and Spatial Binned Color Features

The code for this step is contained in the code cells [4]-[5] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

Color histograms can be helpful for looking at car vs non-car images. I use `np.histogram()` to calculate the color histogram features. Here are color histograms examples of vehicle and non-vehicle, on `YCrCb` color space with 64 bins on each color channel:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Spatially binnned images still retain enough information to help in finding vehicles. I use `cv2.resize()` to calcuate the spatial binned features. Here are spatial binned examples of vehicle and non-vehicle, on `YCrCb` color space with 8x8 pixel resolution on each color channel:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

### Histogram of Oriented Gradients (HOG) Features

The code for this step is contained in the code cells [6]-[7] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

Histogram of Oriented Gradient (HOG) is an effective method to capture the gradient features of various shapes, hence widely used in object detection. I use [`hog()`](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) function in [scikit-image](http://scikit-image.org/) package to calculate the HOG features. Here are HOG examples of vehicle and non-vehicle, on `YCrCb` color space with `orientations=9`, `pix_per_cell=8`, `cell_per_block=2`, and `transform_sqrt=True`:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

### Linear SVC Training

The code for this step is contained in the code cells [8]-[9] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

## Hog Sub-sampling Window Search

The code for this step is contained in the code cells [10]-[12] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

![alt text][image14]
![alt text][image15]

## Video Implementation

The code for this step is contained in the code cells [13]-[16] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

Here's a link to [project video result](./test_videos_output/project_video.mp4).


## Discussion
