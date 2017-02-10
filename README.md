## Vehicle Detection
The goals / steps of this project are the following:
- Extract features using a labeled training set of RGB Images
    - Histograms of Oriented Gradients of each YCrCb color spaces
    - Histograms of value of each YCrCb color spaces
    - Binned raw images of each YCrCb color spaces
* Training a Linear SVM classifier using a extracted features
* Using a sliding-window technique for searching and classifying vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### Histogram of Oriented Gradients (HOG)

#### 1. How to extract HOG features from the training images.
The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `main.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. How to settled on final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. How to trained a classifier using extracted features

I trained a linear SVM using...

### Sliding Window Search

#### 1. How to implemented a sliding window search.  

#### 2. How to decide parameter of window size and overlap

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 3. Show examples of classifier

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Video
Here's a [link to my video result](./project_video.mp4)


#### 2. How to implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

##### Heatmaps of six frames

![alt text][image5]

##### Output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

##### Resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
