# **Traffic Sign Recognition** 
### **John Glancy**


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

[image1]: ./writeup_images/sign_samples.png "Sign Samples"
[image2]: ./writeup_images/sign_frequency.png "Sign Frequency"
[image3]: ./writeup_images/original_pp_images.png "Original, Preprocessed, & Augmented Images"
[image4]: ./writeup_images/new_german_signs.png "New Signs to Test Model"


## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

**Point 1**: Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

> You're reading it! and here is a link to my [project code](https://github.com/thatkahunaguy/USD-P2-Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier_2.ipynb)

### Data Set Summary & Exploration

**Point 1**: Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

> The code for this step is contained in the second code cell of the IPython notebook.  

> I used python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

**Point 2**:  Include an exploratory visualization of the dataset and identify where the code is in your code file.

>The code for this step is contained in the third through fifth code cells of the IPython notebook.  

>Here is an exploratory visualization of the data set. There is a random sample of 25 of the sign images to provide an idea of what they are like.  I then sampled the frequency of images of each sign type as shown in the bar chart.  Note that certain type so signs have a very low number of examples.  I would expect the model will have a more difficult time predicting these signs since there is relatively little data to learn from.  One possible strategy I would attempt with more time would be to augment these low frequency signs more than the other signs with more examples.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

**Point 1**: Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

>The code for this step is contained in the sixth code cell of the IPython notebook.

>As a first step, I decided to convert the images to grayscale as recommended in the Yann Lecunn paper.  I did this by using the Y channel only of a YUV conversion.  I also augmented the data by rotating images a random amount between +/- 15 degrees to generate additional data for the network and make it robust to rotation as recommended by Yann Lecun.  Lastly, I applied a histogram equalization to the image as recommended in the forums.

>Here are some examples of traffic signs before and after grayscaling, rotation, & histogram equalization.

![alt text][image3]

>As a last step, I normalized the image data to a mean of zero (-1 to 1) to improve convergence of the optimizer.

**Point 2**: Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

>The example data provided contained a training, test and validation set so I did not need to split the data.

>My final  augmented training set had 69598 number of images since I created one randomly rotated image for each existing image. My validation set and test set had 12630 and 4410 images respectively.

>The fourth code cell of the IPython notebook contains the code for augmenting the data set with rotation as described previously in 1 above. The images above include an example of rotation augmentation. The difference between the original data set and the augmented data set is that there is a rotated version of each of image in the original set in the augmented set.

**Point 3**: Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

>The code for my final model is located in the seventh code cell of the ipython notebook. 

>My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 grayscale image after preprocessing   							| 
| Convolution1 5x5     	| 1x1 stride, same padding, outputs 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution2 5x5     	| 1x1 stride, same padding, outputs 10x10x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20 				    |
| Flatten conv 2      	| 500 outputs      								|
| Flatten conv 1      	| 1960 outputs      							|
| Fully connected		| 2460 inputs(conv1 + conv2), outputs 200,      |
|						| 0.7 dropout    		                		|
| Fully connected		| outputs 84, 0.7 dropout                       |
| Fully connected output| outputs 43                                    |
| Softmax				| Adam Optimizer						    	|


 
**Point 4**: Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

>The code for training the model is located in the eighth & ninth code cells of the ipython notebook. To train the model, I used the AdamOptimizer to back propogate and minimize a softmax cross entropy function over 10 epochs.  

**Point 5**: Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

>The code for calculating the accuracy of the model is also located in the eighth & ninth code cells of the Ipython notebook.

>My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.1% 
* test set accuracy of 92.6%


>I used an architecture modeled after the one in the Yann & Lecun paper.  It was comprised of 2 convolution/relu/pooling layers with the output of each convolution layer combined as input to the first fully connected layer.  There were a total of 3 fully connected layers prior to output.  This architecture was chosen since it was successful in the Yann & Lecun paper.  
The primary iteration and tuning was done around augmenting and preprocessing the data.  The validation accuracy was continuing to climb slightly and so additional epochs may have continued to improve validation results.  However, since I was travelling without internet access during a portion of this effort and restricted to running on my local machine, I chose to stop at 10 epochs since validation accuracy met the 93% minimum requirement.  I did add dropout layers and tested keep probabilities of 50% and 70% with 70% providing satisfactory results.  I would focus the next step given more time on augmenting the data for signs with few data points before turning to additional model tuning as most of the incorrectly predicted results are for signs with low representation in the training data set.



### Test a Model on New Images

**Point 1**: Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

>Here are five German traffic signs that I found on the web which were shown earlier to illustrate preprocessing:

![alt text][image4]

>The first image might be difficult to classify because because it has a very low number of samples in the training set.  The remainder of the images should be fairly easy to classify as they are well represented in the training set though similar signs such as the speed limit signs may present a challenge.

**Point 2**: Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

>The code for making predictions on my final model is located in the eleventh and twelfth code cells of the Ipython notebook.  Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve      	| Right-of-way at the next intersection  		| 
| Speed limit (30km/h)  | Speed limit (30km/h) 							|
| Speed limit (120km/h)	| Speed limit (120km/h)				    		|
| No entry	      		| No entry					 				    |
| Children crossing		| Children crossing      						|

>For these 5 images the model correctly guessed 4 of the 5 traffic signs which is 80% accurate. This compares favorably to the accuracy on the test set of 93%
The details of the incorrectly identified sign are shown in the thirteenth code cell  

**Incorrect Identification # 1**
   sign:  21   Double curve
was incorrectly identified as:
   sign:  11   Right-of-way at the next intersection

**Point 3**: Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

>The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.  With the exception of the first image, the model was fairly certain of it's predictions(2x or greater probability for the #1 prediction).  The final cell of the notebook prior to the optional step 4 contains the probabilities for each prediction, outlined below.

The predictions for example 0  sign # 21 Double curve  were:
   *probability:  26.31%  Right-of-way at the next intersection
   *probability:  16.59%  Beware of ice/snow
   *probability:  16.57%  Slippery road
   *probability:  11.25%  Dangerous curve to the right
   *probability:  10.58%  Children crossing
 
The predictions for example 1  sign # 1 Speed limit (30km/h)  were:
   *probability:  63.61%  Speed limit (30km/h)
   *probability:  28.03%  Speed limit (50km/h)
   *probability:  2.93%  Speed limit (80km/h)
   *probability:  2.77%  End of speed limit (80km/h)
   *probability:  0.85%  Speed limit (20km/h)
 
The predictions for example 2  sign # 8 Speed limit (120km/h)  were:
   *probability:  83.27%  Speed limit (120km/h)
   *probability:  15.20%  Speed limit (70km/h)
   *probability:  0.91%  No vehicles
   *probability:  0.26%  Speed limit (100km/h)
   *probability:  0.16%  Speed limit (20km/h)
 
The predictions for example 3  sign # 17 No entry  were:
   *probability:  99.93%  No entry
   *probability:  0.07%  Stop
   *probability:  0.00%  Speed limit (120km/h)
   *probability:  0.00%  Turn right ahead
   *probability:  0.00%  Speed limit (20km/h)
 
The predictions for example 4  sign # 28 Children crossing  were:
   *probability:  65.62%  Children crossing
   *probability:  14.68%  Dangerous curve to the right
   *probability:  7.66%  Bicycles crossing
   *probability:  3.54%  Road narrows on the right
   *probability:  2.31%  Slippery road 