Some exercises questions i've done from Chapter 1.) Exercises section of the book:
*Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow*

### Types of Learning Methods, Datasets & more Jargon

**What is a labeled training set? :**
- Dataset where every data entry includes another piece of data describing it.
- This kind of data is used in *Supervised Learning*.
-. It enables ML models to learn the characteristics associated with specific labels.

**What is Supervised Learning?**
- Supervised Learning is when a machine learning algo is trained using labeled training sets. (Data, along with a label describing said data) for every entry.

**What is Unsupervised Learning, and some common applications? :**
- Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets. [IBM ressource](https://www.ibm.com/topics/unsupervised-learning#:~:text=the%20next%20step-,What%20is%20unsupervised%20learning%3F,the%20need%20for%20human%20intervention.) 

**What is Semi-Supervised Learning, and give an application :**
- Semi-supervised learning is when the algorithm is trained using both *Labelled* and *Un-Labelled* data.

- **EXAMPLE:** Google Photos, when you upload photos to google, they can recognize people, animals and other specific elements within the picture. These photo uploads would be the *unsupervised* part of the data (un-labelled). **Now,** in Google Photos, you can select a section of the imagine and *label* people & pets, these would be the labelled instances of the data. Labelling an area on a picture as `Jason, a human male`. Google Photos can identify people very efficiently, thanks to the unsupervised portion; Massive amount of data they have on the physical appearance of humans.
-  And it can also recognize me personally, Jason, thanks to the labelled data points, from here it can check if there are any pictures of `Jason` and `Xavier` *(some other labelled data points corresponding to my friend Xavier perhaps)* on my Google Photos drive.

**What's Online Learning? :**
- Online Learning allows a Machine Learning System to learn on data whilst preforming actions on it *(Learning as the Net goes)*
- Online Learning works best when your Neural Network needs to evolve with the data it's being passed. (Example: A neural network that needs to stay trendy, and hip)
- A.K.A **Incremental Learning.**

**What's Offline Learning? :**
- Offline Learning is when a Neural Network is trained initially, once in production and classifiying un-seen data, this new data wouldn't affect the current model. This is good when you know a **model** is working well for it's use, and you don't want it to get skewed by new data.
- A.K.A **Batch Learning.**

**What is out-of-core Learning? :**
- This is when you're training an ML algorithm on a <b>HUGE</b> dataset, and can't fit all the data on your machine, so you extracts parts of the Data (from the cloud, TCP, whatever...) runs a training step with the small extracted portion of data, then redo the process with the next bit of data. This way you can use massive datasets to train Neural Nets even if you have limited data storage.

**What's Instance Based & Model Based *Machine Learning?* :**
- Instance Based ML is when a new and unknown datapoint is classified by it's surroundings data points which are known, it basically takes a similarity measure to all surrounding entries *(In terms of positon on the graph in this case,but more boardly in some vector space)*, then from here, based on all it's findings, will calculate some kind of mean similarity value, then associate to the most similar classification.
![Instance Based](./static/img/instance_based_ml.png)

- Model Based ML is when the training data is modelled mathematically, and has a very specific definition of what is, and what is not. Think of Model Based ML to have a continuous perimeter arround certain datapoints, *modelling* that region, whereas Instance Based works with more relative, similar and discrete points instead.
- Essentially, the Model for the Data is generated once, and whenever unknown datapoints need to get classified, it can very quickly view if the entry fits the model or not. Or in more complex applications, which model does it fit best? *(In this case it's the stars section of the chart)*
![Model Based](./static/img/model_based_ml.png)

### Some Applications:

**2 Most Common Supervised Learning Tasks? :**
> 1.) Spam Classification
- Since it's either Spam or Not Spam (Spam or Ham as the cool kids say), it's easy to create datasets with various emails, along with their determined type (To be, or not to be spam)

>2.) Classification & Regression problems in general
- Since this kind of task is trained to spit out a very specific answer (It's a spam email for the example above, or It's the number 5, *MNIST*) Supervised Tasks thrive on Labelled Data, since this allows it to distinguish different data points, and preform regression techniques for modelling the problem, and where to draw the line between say a number *3* and an *8*.

**What techniques would be useful to train a neural network for a Robot that walks on wild/natural terrain that's uncertain? :**
- In terms of training the machine learning algo based on visualizing the outside terrain in vector space, natural terrain is quite random in the sense that it's shape & arrangement is very unpredictable, example: It's very hard to simulate an "average" river, even if we got an avg. *width, depth, speed of running water* of many rivers, we'd still need to determine it's flow through an environment based on previous factors, then simulate how another randomly generated element of the terrain is simulated based on the previous layer... *(another ex. trees, that tend to grow near the river)* This seems very complex and **unfeasible** ...
- ^ Due to the terrain's highly-complex (seemingly random) pattern, I think **Online Learning** is a huge must for this task, this would improve the robot drastically since with more data, this complex problem now has more data-points so hopefully this allows us to anticipate the terrain to a higher degree. (more accurately)
- If we're going with Online Learning, the live data we're feeding to the training model is going to be unlabelled, since we can't label the data on the fly unfortunately. (therefore this task must be an *Online & Unsupervised learning task*) The neural network would still be able to look at other parameters such as Geo-location *(Maybe the model knows that the Terrain in Peru near [**Machu Picchu**](https://www.peruforless.com/blog/machu-picchu-mountain/) is very Mountainous, Rocky & Ridged)*
- I think the last technique i'd apply for this task is to also use some kind of metric of success & apply a re-enforcement task to help the task to evolve in the right direction. This metric could be a measure of how long it takes the average human to preform the walk, versus the robot `Human Travel Time / Robot Travel Time` for example. from here I'd know the bot preforms poorly if the human takes 10 minutes and the robot takes 20. (10/20) = ~50% Success

**TL:DR ->** Online & Unsupervised Learning Task, that tries to optimize for success, and learns from it's successes.
 
