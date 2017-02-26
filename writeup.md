#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/training_data_graph.png "Visualization"
[image2]: ./examples/preprocessing.png "Preprocessing"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/test_images/50limit.jpg "Traffic Sign 1"
[image5]: ./examples/test_images/bumpy_road.jpeg "Traffic Sign 2"
[image6]: ./examples/test_images/end_no_passing.jpg "Traffic Sign 3"
[image7]: ./examples/test_images/fahrverbot.jpg "Traffic Sign 4"
[image8]: ./examples/test_images/keep_right.jpg "Traffic Sign 5"
[image9]: ./examples/test_images/priority.jpg "Traffic Sign 6"
[image10]: ./examples/test_images/slippery_road.jpg "Traffic Sign 7"
[image11]: ./examples/new_images_predictions.png "New Image Predictions"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used python to display summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. I have created 3 bar charts to se how the data is distributed between the bins in each train, validation and test datasets.
This has showed me that the data is not evenly distibuted and this might cause future bias towards the most numerous classes.
This iequality might have to be address if the model performance was poor.

I have also printed a sample image from each class. This has shown me that some of the images are really hard to read. 
This has prompted me to explore options with histogram normalization.

I have experimented with equalizing histogram which was not satysfying as some images had a big difference in ligting levels.
This has made me use adaptive histogram normalization. I have experimented with various clip limits and grid sizes. 
I settled for 3 by 3 grid and clip limit of 12. I was torn between 3 and 4 but I think 3 gives better results.

I decided against converting not to loose too much information. The problem is small enough that there was really no reason that would convince me to do it.

Another thing that I was debating was to detect edges on each of the channels. That might possibly facilitate training as there would be less data on each image to process. But when I tried it the model actually took longer to train and performed worse.

Here is the distribution of the training data.

![alt text][image1]

Driven by the review I decided to convert to grayscale, but I must say that I can't see a drop in training time or final accuracy.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I'm runnning adaptive histogram equalization on each color channel separately.

Here is an example of some processing I've considered:

![alt text][image2]

From left:
- original image,
- image after a global histogram equalization,
- image after separate canny detection on each channel
- image displayed with a HSV colormap
- image after global canny detection
- image after separate adaptive histogram equalization on each channel
- image after separate adaptive histogram equalization on each channel converted to grayscale

As a last step, I normalized the image data because it makes training more efficient.
The goal of normalization is to bring the data to zero mean and equal variance.
This is to make sure that the comuted gradients are not afected by the changes in range between the images.
In case of these images it effictively meant to get rid of the differences of the lighting across the images.
Achieving equal variance is effectively accomplished by the adaptive histogram equalization. A good point to make here is that this worked a profoundly better than min-max scaling.
So the normalization step is used here for achieving zero mean and changing the data type so that the gradients of all the images are comparable to each other. This step has also dramatically increased the learning speed.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The dataset was already split into train, validation and test sets.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

I did not generate additional data.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My model was just a slightly modified LeNET and consisted of the following layers:

| Layer					|	Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x3 RGB image   							| 
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x18	|
| RELU					| with dropout									|
| Max pooling			| 2x2 stride,  outputs 14x14x18 				|
| Convolution 3x3		| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					| with dropout									|
| Max pooling			| 2x2 stride,  outputs 5x5x16					|
| Fully connected		| Flattened 400 to 120							|
| RELU					| with dropout									|
| Fully connected		| 120 to 84										|
| RELU					| with dropout									|
| Fully connected		| 84 to 10										|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the tenth cell of the ipython notebook. 

To train the model, I used Adam optimizer.
During training the RELUs have dropout set to 0.5.
I've put L2 regularization on all the weights and biases with 0.000001 ratio.
The learning rate starts off with 0.01 to speed up initial learning and then gets dropped to 0.00001 after crossing validation accuracy of 0.7.
I have a hard stop for training after reaching 0.96 accuracy so the training does not take longer than necessary.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.965 
* test set accuracy of 0.949

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I've started with basic LeNET model as I had it ready from the previous lab.
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I had to change the initial layer because I wanted to process all 3 channels of the input images.


* Which parameters were tuned? How were they adjusted and why?
My original dropout was set to 0.8. Dropping it to 0.5 gave me better results on the test set.
At first I used large batch size (128 or 256) thinking it would be faster to train as long as it fits memmory. It turned out that it required a lot of epochs. When I lowered the batch size to 64 I got to 0.8 accuracy after the first epoch.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I think that convolutions work well here, because they can abstract features like lines and colors. Dropout helped with overfitting so that the model performed well on test data

If a well known architecture was chosen:
* What architecture was chosen?
LeNET
* Why did you believe it would be relevant to the traffic sign application?
It seemed to be deep enough.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The fact that the test accuracy was close to validation accuracy and the new image accuracy being high tells me that the model can abstract the problem well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] 

Two of the images are italian with a thicker border. The last image is rotated slightly. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The predictions are run in cell 15 and analyzed in cell 16.
The accuracy of the predictions differs from run to run between 70-100% . Suprisingly the bumpy road sign is the sign that gets missclassified along with the rotated slippery road sign. Both belong to one of the least populated bins in the training set.


Here are the results of the prediction (included the least accurate run on the final model):

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 50		| Speed limit 50								| 
| Bumpy road 			| Bumpy road									|
| End of no passing		| End of no passing								|
| No vehicles			| No vehicles					 				|
| Keep right			| Keep right									|
| Priority road			| Priority road									|
| Slippery road			| Bicycles crossing								|


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the predictions visualized:

![alt text][image11] 

As you can see the predictions on all the classes other than the slippery road sign are very close to 1. This means that for most the bins the model is very certain of the answer. The second candidates have probabilities few order of magnitiude smaller.
In case of the incorrectly predicted sign the correct sign is classified as third and the probabilities of the 4 first classes are very close. Again if you look at the distribution of the examples it shows that these are the classes with the fewest training examples.
That brings me to a conclusion that the right next step would be to generate more examples.
We could do that by using combination of few transformations like noise, translation, skew and rotation.
