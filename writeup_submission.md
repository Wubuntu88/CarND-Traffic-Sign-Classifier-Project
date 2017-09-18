# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sign_pictures]: ./plots/sign_pictures.png "class pictures"
[classes_histogram]: ./plots/classes_histogram.png "classes histogram"
[loss_graph]: ./plots/loss.png "loss graph"
[accuracy_graph]: ./plots/accuracy.png "accuracy graph"

[tf1]: ./GermanTrafficSigns/AheadOnly.jpeg "Traffic Sign 1"
[tf2]: ./GermanTrafficSigns/GeneralCaution.jpeg "Traffic Sign 2"
[tf3]: ./GermanTrafficSigns/NoEntry1.jpeg "Traffic Sign 3"
[tf4]: ./GermanTrafficSigns/NoEntry2.jpeg "Traffic Sign 4"
[tf5]: ./GermanTrafficSigns/Pedestrians.jpeg "Traffic Sign 5"
[tf6]: ./GermanTrafficSigns/PriorityRoad.jpeg "Traffic Sign 6"
[tf7]: ./GermanTrafficSigns/PriorityRoad2.jpeg "Traffic Sign 7"
[tf8]: ./GermanTrafficSigns/RoundaboutMandatory.jpeg "Traffic Sign 8"
[tf9]: ./GermanTrafficSigns/RoundaboutMandatory2.jpeg "Traffic Sign 9"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* I calculated summary statistics of the data set:
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is a list of pictures for each class in the dataset:

![alt text][sign_pictures]

Here is histogram of the number of pictures in each class in each of the data partitions (training, validation, and test).

![alt text][classes_histogram]

# Description of Histogram
 The histogram above shows the number of pictures that belong to each class. Some classes have few pictures (under 250), and some have many (close to 2000). The histograms of the training, test, and validation data have the same shapes, meaning that records for a given class were proportionally distributed accross the data sets.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to normalize the image with the following function:
```python
def normalize_image(image):
    """
    @param: image (Numpy array representing an image)
    Note: 128.0 is the midpoint used for normalization
    """
    return (image - 128.0) / 128.0

X_train = normalize_image(X_train)
X_valid = normalize_image(X_valid)
X_test = normalize_image(X_test)
```

The pixel values range from 0 to 255.  In this normalization, the new range is \[-1, 1).  The reason for this normalization is for the algorithm having an easier time dealing with the data.  Because the range is smaller, the floating point values can have better precision.  Because all of the images have the same range of values, normalization is not as beneficial as when the variable are of different ranges.
I also tried a normalization where the new range would be \[0, 1), but that did not seem to give any better results, so I stuck with a range of \[-1, 1).
I did not turn the image to greyscale.  The reason is that I thought there would be too much data loss, because the colors in traffic signs are important.  However, I did not try greyscale so I do not know if this is the case.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x32 	|
| RELU					|												|
| Fully connected layer		|         									|
| RELU					|												|
| DROPOUT					| keep_probability = 75% |
| Fully connected	layer	|         									|
| RELU					|												|
| DROPOUT					| keep_probability = 75% |
| Fully connected		|         									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used an Adam Optimizer to train the model.  It was used in the LeNet model so I kept it.  It is supposedly more sophisticated than plain SGD.  I thought that if the LeNet model used it, it would be a good optimizer.

The batch size I chose was 128.  For some reason, when I chose a larger batch size, my performance went down.  I'm not sure why, and I feel like that should not happen.

The number of epochs was 200.  In general, the longer I trained it, the better my results were.  The strange thing is that the loss would get gradually worse for each epoch, but suddenly drop significantly.  Another strange thing is this sharp dropoff was not apparent in the accuracy.

The learning rate was set to .0001.  I played around with the learning rate quite a bit.  I decided on a slow learning rate because that gave me a more gradual curve.  A slower learning rate is better, and I had a GPU and time, so I just kept the slow learning rate.  Also, faster learning rates did not seem to end up with a great result.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

#### training_accuracy: 0.999
training_loss: 0.00224

#### validation_accuracy: 0.934
validation_loss: 0.445

#### test_accuracy: 0.93
test_loss: 0.602

17, 35, 18, 17, 40, 27, 40, 12, 12  <-- actual labels
12, 12, 11, 40, 11, 12, 11, 12, 11 <-- predicted labels
#### accuracy on images from the web: 0.111

#### Note: on other runs of my algorithm, I would get validation accuracy results of around 9.4, but on my most recently saved network, the result is 0.934.

#### Here are two graphs that represent the loss and accuracy of my network through the epochs.  A horizontal line is drawn at .93 to show the goal accuracy on the second graph.
![alt text][loss_graph]

![alt text][accuracy_graph]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
#### I first tried the LeNet architecture that was used in the tensorflow lab.  I chose it because I could easily use the material from the lab, and it was easy to understand the architecture.
* What were some problems with the initial architecture?
#### The architecture did not alone achive good enough accuracy to put it over the .93 threshold.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 I will discuss pooling, dropout, and the activation functions.
#### Pooling
 I decided to remove pooling because I thought that it would diminish my data at each convolutional layer, and that change would not benefit the accuracy.  When I compared my solution with and without pooling, pooling tended to give my network worse accuracies, so I got rid of that component.
#### Dropout
  I tried my model with and without dropout several times.  From what I saw, it seemed like dropout gave worse results when I had fewer epochs, but the same results when the number of epochs was high (around 200).  This makes sense, because dropout makes several of the weights inactive at the layer, and unable to adjust at a given epoch.  Thus, it is natural that it would take longer for the network to train.  Eventually with enough epochs, the network gave comarable accuracies with dropout compared to without.  I decided to keep dropout because I was willing to wait the extra time.
#### Activation Functions.
  I decided to use relus after each layer, as that was the recomended activation function.  I steered clear of the softmax function for the terminal activation function and did not have a terminal activation function.  I had not know that this was possible, but I saw other examples leaving out the terminal activation function, so I followed suit.  I tried the softmax terminal activation function several times and got horrible results, so that is another reason I left it out.
* Which parameters were tuned? How were they adjusted and why?
#### Learning Rate
I experimented with many learning rates.  I finally arrived at the learning rate of 0.0001.  I found that higher learning rates would max out the accuracy quickly and at a lower level that the slower learning rate.  I compared the loss curves and setted on this learning rate because the loss curve was a more gradual slope downward than higher learning rates.  This low learning rate also gave relatively good accuracy after many epochs.  The loss curve in the graph above does seem steep - too steep in fact.  But remember that I use many epochs, so the curve looks steeper than it should.  This learning rate gave a much more gradual loss curve than higher learning rates.
#### Dropout's Keep Probability
 I chose a keep_prob for the dropout of 75%.  This number was somewhat arbitrarily chosen.  I did not go lower because I didn't want the network to take too long to train; I did not go higher because I thought that the higher one chooses the probability, the more that defeates the purpose of doing dropout.
#### Number of Filters
 I chose to have 32 filters for each layer of the convnet.  The rationale for the number 32 was that I wanted as many filters as possible without the model taking forever to finish.  32 ended up being a good number because it allowed me to reach the .93 threshold, and it reached it in a reasonable amount of time.  I had tried 64 filters; that took a long time and did not give me better results than the 32 filter model.
  The LeNet model increases the number of filters for each convolutional layer.  I tried increasing and decreasing the number of filters for each layer, as well as keeping them the same, and there was not much difference, so I just kept the filters the same for each convolutional layer.
#### Convolution Size and Stride
 I chose a 5x5 convolution and a stride of 1.  The reason for choosing the 5x5 convolution was because it is a standard convolution size and I thought it would give good results.  I chose the stride length of 1 because I did not want to loose too much information by downsampling the image aggressively with a higher stride.
#### Padding
 I chose valid padding because it was used in LeNet, and I didn't want to have to deal with recomputing the dimensions of the network with the other padding type.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
#### I chose to base my model on Lenet because it easy to modify and does a fairly good job at image classification.
* Why did you believe it would be relevant to the traffic sign application?
#### I do not belive that the model is specifically relevant to traffic signs, just images in general.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
#### The modified model does achieve an accuracy on the validation set above the threshold value (.93), and the accuracy on the test set is not bad (0.93).
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][tf1] ![alt text][tf2] ![alt text][tf3] 
![alt text][tf4] ![alt text][tf5] ![alt text][tf6]
![alt text][tf7] ![alt text][tf8] ![alt text][tf9]

 Difficulties classifying these images may be due to several reasons.  First, some of the images are not centered in the picture, and there are irrelevant colorful backgrounds.  Second, there are Getty Images labels over some of the signs, which degrades the image's integrity.  Third, I had to resize the images.  This makes some of the images look awkward and have possibly different scales than the images in the training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
[17, 35, 18, 17, 40, 27, 40, 12, 12]  <-- actual labels
[17, 13, 18, 17, 40, 18, 40, 12, 12] <-- predicted labels
accuracy on images from the web: 0.778

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|No entry|No entry
|Ahead only|Yield
|General caution|General caution
|No entry|No entry
|Roundabout mandatory|Roundabout mandatory
|Pedestrians|General caution
|Roundabout mandatory|Roundabout mandatory
|Priority road|Priority road
|Priority road|Priority road


The model was able to correctly guess 7 out of the 9 traffic signs.  This is decent classification accuracy.  Previously I had been getting results of 0% accuracy, but that was because when the jpegs were loaded, they were loaded with a range of 0 to 1, and my image normalization was all wrong.

The AheadOnly sign was misclassified as Yield.  These signs are quite different.  The only reason that I can think of that would cause this is that there is a large, colorful background in the AheadOnly sign.  That perhaps gave a incorrect classification.

The Pedestrians sign was misclassified as a general caution.  This is understandable because the signs look alike; both are triangles with red border, white inside, and a black image.  I am not sure how to fix this misclassification other than to have a better classifier. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The summary predictions are as follows:
### No Entry (correct) | (picture 1)
probabilities: \[     1.0, 1.83e-19, 7.23e-25, 2.25e-26, 1.93e-26]

predictions:   \[      17,       14,       25,       28,       32]

labels:        \[No entry, Stop    , Road work, Children crossing, End of all speed and passing limits]

### Ahead Only ; but Yield was predicted (incorrect) | (picture 2)
probabilities: \[   0.736,    0.141,   0.0722,    0.041,  0.00531]

predictions:   \[       2,       22,       40,       39,        8]

labels:        \[Speed limit (50km/h), Bumpy road, Roundabout mandatory, Keep left, Speed limit (120km/h)]

### General Caution (correct) | (picture 3)
probabilities: \[     1.0, 2.43e-20, 1.08e-20, 4.82e-23, 3.49e-25]

predictions:   \[      18,       37,       40,       26,       11]

labels:        \[General caution, Go straight or left, Roundabout mandatory, Traffic signals, Right-of-way at the next intersection]

### No Entry (correct) | (picture 4)
probabilities: \[     1.0, 1.05e-13, 5.15e-14, 4.06e-15, 2.83e-16]

predictions:   \[      17,        1,        0,       30,       13]

labels:        \[No entry, Speed limit (30km/h), Speed limit (20km/h), Beware of ice/snow, Yield   ]

### Roundabout Mandatory (correct) | (picture 5)
probabilities: \[     1.0, 2.56e-07, 5.59e-09, 2.17e-09,  1.9e-09]

predictions:   \[      40,        7,       11,        2,       42]

labels:        \[Roundabout mandatory, Speed limit (100km/h), Right-of-way at the next intersection, Speed limit (50km/h), End of no passing by vehicles over 3.5 metric tons]

### Pedestrians; but No Passing was predicted (incorrect) | picture 6
probabilities: \[     1.0, 3.64e-09, 7.82e-14, 3.93e-14, 3.85e-14]

predictions:   \[       9,       30,       20,       10,       18]

labels:        \[No passing, Beware of ice/snow, Dangerous curve to the right, No passing for vehicles over 3.5 metric tons, General caution]

### Roundabout Mandatory (correct) | (picture 7)
probabilities: \[     1.0, 6.18e-19,  2.9e-19, 7.36e-22, 5.94e-22]

predictions:   \[      40,       37,       38,        2,       39]

labels:        \[Roundabout mandatory, Go straight or left, Keep right, Speed limit (50km/h), Keep left]

### Priority Road (correct) | (picture 8)
probabilities: \[     1.0,  1.7e-29,      0.0,      0.0,      0.0]

predictions:   \[      12,        6,        0,        1,        2]

labels:        \[Priority road, End of speed limit (80km/h), Speed limit (20km/h), Speed limit (30km/h), Speed limit (50km/h)]

### Priority Road (correct) | (picture 9)
probabilities: \[     1.0, 4.12e-07, 7.39e-08, 5.01e-11, 2.48e-11]

predictions:   \[      12,       26,       15,       16,       25]

labels:        \[Priority road, Traffic signals, No vehicles, Vehicles over 3.5 metric tons prohibited, Road work]

### Insights
The confidence of each prediction is very high.  Almost all of them are shown to be one (a perfect one because they are rounded up).  The only image not one is picture 2, which is 0.736 confident, and was incorrectly classified.  It is a good thing that the incorrectly classified image was not classified correctly.  In fact, the real class is not in the top 5, leading me to think that the colorful background of that image makes it hard to detect.
The Other image that was incorrectly classified was the 'Pedestrians' sign misclassified as 'No Passing'.  Both are signs have red borders and white in the middle.  One disturbing fact is that it was misclassified as 'No Passing' with 100% confidence using softmax, but misclassified as 'General Caution' without softmax.  This makes me think that I did something wrong using TensorFlow, but I am not sure what.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


