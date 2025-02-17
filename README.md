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

[image0]: ./output_images/example_car_noncar.png "Data Visualization"
[image1]: ./output_images/false_positives.png "False Positives"
[image2]: ./output_images/color_hist_car1.png "Color Histograms Car 1"
[image3]: ./output_images/color_hist_car2.png "Color Histograms Car 2"
[image4]: ./output_images/color_hist_noncar1.png "Color Histograms Not-car 1"
[image5]: ./output_images/color_hist_noncar2.png "Color Histograms Not-car 2"
[image6]: ./output_images/color_bins_car1.png "Spatial Bins Car 1"
[image7]: ./output_images/color_bins_car2.png "Spatial Bins Car 2"
[image8]: ./output_images/color_bins_noncar1.png "Spatial Bins Not-car 1"
[image9]: ./output_images/color_bins_noncar2.png "Spatial Bins Not-car 2"
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

![alt text][image0]

To reject the false positives, hard negative mining was adopted. I collected 1473 [non-car images](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/training_data/dataset_falsePos.p) from the project video, adding them to the training set and finally retraining the classifier. Here are several sample false positive images:

![alt text][image1]


### Color Histograms and Spatial Binned Color Features

The code for this step is contained in the code cells [4]-[5] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

Color histograms can be helpful for looking at car vs non-car images. I use `np.histogram()` to calculate the color histogram features. Here are color histograms examples of vehicle and non-vehicle, on `YCrCb` color space with 32 bins on each color channel:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Spatially binned images still retain enough information to help in finding vehicles. I use `cv2.resize()` to calculate the spatial bin features. Here are spatial bin examples of vehicle and non-vehicle, on `YCrCb` color space with 32x32 pixel resolution on each color channel:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

### Histogram of Oriented Gradients (HOG) Features

The code for this step is contained in the code cell [6] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

Histogram of Oriented Gradient (HOG) is an effective method to capture the gradient features of various shapes, hence widely used in object detection. I use [`hog()`](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) function in [scikit-image](http://scikit-image.org/) package to calculate the HOG features. Here are HOG examples of vehicle and non-vehicle, on `YCrCb` color space with `orientations=9`, `pix_per_cell=8`, `cell_per_block=2`, and `transform_sqrt=True`:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

### Linear SVC Training

The code for this step is contained in the code cells [7]-[9] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

Several combinations of color space, color and HOG parameters were explored, and the final values are summarized as follows: 

| Parameters        | Value   | 
|:-----------------:|:-------:| 
| Color space       | YCrCb   | 
| Color hist bins   | 32      |
| Spatial bin size  | (32, 32)|
| 'orientations'    | 9       |
| 'pix_per_cell'    | (8, 8)  |
| 'cell_per_block'  | (2, 2)  |

Feature extraction was performed on 17760 samples in the dataset, including 8792 cars and 8968 non-cars. Then, the total 8460 features, which consist of 96 color histogram features, 3072 spatial bin features, and 5292 HOG features, were normalized using [`sklearn.preprocessing.StandardScaler()`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) method. The 17760 normalized samples were shuffled and split into training and test sets:

```python
X_train, X_test, y_train, y_test = train_test_split(scaled_X, labels, test_size=0.2, stratify=labels)
```

where `stratify=labels` ensures that the ratios between cars and non-cars are the same in both training and test sets.

A linear SVM classifier is built and trained using [`sklearn.svm.LinearSVC()`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). To tune the SVM with a linear kernel, I consider five possible values of the `C` parameter, and the results are summarized in the following table:

| C     | Test accuracy [%]   | Training accuracy [%] | Training time [s] |
|:-----:|:---------------:|:-----------------:|:-------------:| 
| 1     | 98.85       | 100       | 25.9    | 
| 0.1   | 98.85       | 100       | 26.1    |
| 0.01  | 98.87       | 100       | 26.6    |
| 0.001 | 98.93       | 100       | 9.5     |
| 0.0001| 98.82       | 99.63     | 4.9     |

`C=0.001` is chosen so that the classifier is more generalizable. Having chosen the features and SVM hyperparameters, I retrained the classifier on the entire 17760 training samples, plus 1473 false positive samples. Finally, 99.99% re-train accuracy is achieved under a large regularization term (`C=0.001`).

## HOG Sub-sampling Window Search

The code for this step is contained in the code cells [10]-[12] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

An efficient window shearch approach is adopted, which allows to extracts hog features once (per scale value) and then preform sub-sampling in each window. Multi-scaled search windows are used, whose parameters are shown below:

| Scale | Window Size | y Range   | Overlap |
|:-----:|:-----------:|:---------:|:-------:| 
| 1     | (64, 64)    | (400, 500)| 0.875   | 
| 1.25  | (80, 80)    | (400, 580)| 0.75    |
| 1.5   | (96, 96)    | (400, 660)| 0.75    |

To reject false positives, an effective way I found is to use [`clf.decision_function()`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function) to predict confidence scores for samples, rather than using `clf.predict()` to predict labels. Similar to the notion of probability, one can customize the positive detection threshold value, in the sense that only high-confidence detection (like 80% probability) is accounted for.
In addition, heat-map is used to record multiple detections in an image, and only the "hot" parts of the map are considered as high-confidence detections. Two threshold values for score and heat-map are chosen as follows:

| Parameter   | Threshold  |
|:-----------:|:----------:| 
| Score       | 1.0        |
| Heat (individual image) | 2 |

Once a thresholded heat-map is obtained, I use `scipy.ndimage.measurements.label()` to put bounding boxes and labels around the detected vehicles. The following two examples show the window search and classification on two test images:

![alt text][image14]
![alt text][image15]

## Video Implementation

The code for this step is contained in the code cells [13]-[16] of [`Vehicle Detection.ipynb`](https://github.com/YuxingLiu/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb).  

Similar to the [Lane Finding Project](https://github.com/YuxingLiu/CarND-Advanced-Lane-Lines), a `Vehicle()` class was defined to improve the robustness of the vehicle detection algorithm, by enabling the following methods:
1. Accumulate the heat-map over n past frames to get stable bounding boxes.
2. Apply threshold to the accumulated heat-map to remove false positives.

| Parameter     | Value |
|:-------------:|:-----:| 
| Num of frames | 15    |
| Heat (video)  | 30    |

Here's a link to [project video result](./test_videos_output/project_video_v8.mp4).


## Discussion

Eventually, I found that the following methods are quite effective to reject the false positives:
1. Smaller `C` values for linear SVM classifier to prevent overfitting and to improve generalization. 
2. Hard negative mining.
3. Confidence scores instead of labels in each window search.
4. Multi-frame accumulated heat-map with relatively high threshold.

One effective way to further improve the performance of the classifier is to augment the training data by using the [Udacity dataset](https://github.com/udacity/self-driving-car/tree/master/annotations). It is also possible to use convolutional neural network to avoid manual feature selection. 

Currently, the pipeline runs at 0.5 FPS. To improve the computation speed, some improvements could be helpful:
1. Vectorize the confidence scores prediction.
2. Apply [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) to remove the redundant features.
3. Integrate the lane finding and vehicle detection algorithm to restrict the search region. For instance, if the vehicle is on the left lane (solid yellow line on left), only search the right side of the yellow line.
