**Vehicle Detection Project**

If you're just interested in the result video, [click here!](https://youtu.be/GN0yOMfmp7c)

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier: Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg "Car vs Not Car"
[image2]: ./output_images/HOG_example.jpg "HOG Examples"
[image3]: ./output_images/all_windows.jpg "All Sliding Windows"
[image4]: ./output_images/heat_map.jpg "Heat Map"
[image5]: ./output_images/hot_windows.jpg "Hot windows"
[image6]: ./output_images/heat_map_labels.jpg "Heat Map Labels"

[video1]: ./detection_output.mp4 

---
###Writeup / README  

**Histogram of Oriented Gradients (HOG)**

The function I used to extract HOG features is called get_hog_features and it is located in the first block of the P5 Vehicle Detection.ipynb jupyter notebook under the "Set up initial functions" header.  

I started by exploring on individual images before reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example that shows the output of some tinkering I did with the parameters on different color spaces:

![alt text][image2]

I noticed that the V-channel was producing something extremely similar to the gray scaled image, so I assume the conversion did something that is very much akin to gray scaling, and I determined it was not going to be useful to me. The image I was most concerned with is titled "Gray 8 Orient, 8 PPC, 2 CPB". The abbreviations stand for Orientations, Pixels per cell, and cells per block respectively. I noticed that the hog features picked up the horizontal base of the vehicle really well, as well as the sloped top corners. These features are pretty universal across all cars, and that helped in my decision to use 8 orientations, 8 PPC, and 2 CPB.
I included an image of the same hog result on an image of a non-car for comparison reasons. Clearly there are no horizontal lines that would denote the base of a car. 


**Training the Classifier**

The code pertaining to the training of the classifier is in the notebook under the heading "Train the classifier". Firstly, I verified that there was roughly an equal number of vehicle and non-vehicle images to train on. The result of this check is located in the workbook directly above the same header. I decided to make use of as many features as possible, but realized that that would create a huge feature vector length. I decided I could lose some spatial size for my spatial features, as the lessons pointed out: the pixelized car was still identifiable. I also lowered the number of hist_bins to 18 as I felt that was an adequate number to identify a vehicle. At first I was using the 'HSV' color space, but I was able to achieve better test accuracy by using 'YCrCb'. I extracted all the car features and non-car features and stacked them together so I could split them into a testing set and a training set. I chose a test set size of 20%. I then scaled the feature vector using a StandardScaler to prevent any one particular set of features dominatiing the decision making. I then fit the classifier, which took 12.4 seconds for a feature vector length of 5526. I am not entirely satisfied with the feature vector length, as I think it is too long - but I like the high accuracy it provides. 
The classifier had a test set accuracy score of 0.992 (99.2%), which is pretty impressive.


**Sliding Window Search**

The function that creates the sliding windows is called get_windows, which makes use of slide_window. Both functions are in the functions block in the notebook. I decided to have three distinct groupings: a small section for cars far away, a large comprehensive section in the middle that would hopefully identify the majority of the vehicles, and one group for vehicles that are very close and large in the image. I wanted to keep the number of windows as small as possible - and I think I can definitely go lower than this. Below is a visualized output of my search boxes painted onto an image.

![alt text][image3]

**Image pipeline**

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]

One extra step I took to eliminate false positives was to threshold my search_windows function by using the decision_function described in the scikit learn documentation here: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function
What this allowed me to do was to limit my "hot windows" result to windows where there was a higher degree of confidence in the positive detection. This way, the algorithm is less likely to identify a patch of road as a car.

---

### Video Implementation

Here's a [link to my video result](https://youtu.be/GN0yOMfmp7c)


**Heat map and elimination of false positives**

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. The code where this takes places is under the header "Image Pipeline" in the notebook. I kept a "memory" of 40 frames, and required there to be at least 30 detections in order to be classified as a positive result. It's important to note that this threshold can be reached in way less than 30 frames if there are multiple detections on the same vehicle in different scales.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from an individual image. It glows brighter where there were multiple detections across neighboring windows:

![alt text][image4]


As more frames contribute detections, the "hot spots" continue to stay bright. I added a "Subtract heat" function initially in an attempt to reduce heat where there may have been a false positive, but ultimately scrapped this idea as heat thresholding and the decision_function were doing an adequate job of eliminating false positives.


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]

The false positives on the left of the image would be eliminated via the heat map thresholding.

---

**Discussion**

I had a lot of problems with my sliding windows. I wanted to maximize detections while simultaneously minimizing sliding windows. Eventually I gave in and started to increase overlap between windows to ensure the car would be detected, but this resulted in many windows and slow processing of the video. This pipeline fails on vehicles that are eclipsed by other vehicles, despite the fact that a driver would know that there is still a car behind the car closest to the driver. The pipeline could also fail on vehicles such as trucks or anything with a trailer. It also occurred to me that I never checked to see if the training set had the front view of cars, or if this pipeline is useless for oncoming traffic. I think a more robust approach could be check for anything that isn't road, as we know what road is supposed to look like. Once we know we have something that isn't road, we can start to heat map it and track it across frames. I think an approach where we focus our region of interest to detected objects would be more efficient as well. 
The real struggle with this project was finding a balance between number of sliding windows, while eliminating false positives. The pipeline would often perform admirably on my test videos which were quicker to process, but then I'd notice some failures after processing the whole video, and I'd have to start from scratch.  

