Some exercises questions i've done from Chapter 1.) Exercises section of the book:
*Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow*

**What is a labeled training set? :**
- Dataset where every data entry includes another piece of data describing it.
- This kind of data is used in *Supervised Learning*.
-. It enables ML models to learn the characteristics associated with specific labels.

**2 Most Common Supervised Learning Tasks? :**
> 1.) Spam Classification
- Since it's either Spam or Not Spam (Spam or Ham as the cool kids say), it's easy to create datasets with various emails, along with their determined type (To be, or not to be spam)

>2.) Classification & Regression problems in general
- Since this kind of task is trained to spit out a very specific answer (It's a spam email for the example above, or It's the number 5, *MNIST*) Supervised Tasks thrive on Labelled Data, since this allows it to distinguish different data points, and preform regression techniques for modelling the problem, and where to draw the line between say a number *3* and an *8*.

**What is Un-Supervised Learning, and some common applications?:**
- Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets. [IBM ressource](https://www.ibm.com/topics/unsupervised-learning#:~:text=the%20next%20step-,What%20is%20unsupervised%20learning%3F,the%20need%20for%20human%20intervention.) 

**What's online Learning?**
- Online Learning allows a Machine Learning System to learn on data whilst preforming actions on it *(Learning as the Net goes)*
- Online Learning works best when your Neural Network. 

**What's offline Learning?**
- Offline Learning is when a Neural Network is trained initially, once in production and classifiying un-seen data, this new data wouldn't affect the current model. This is good when you know a **model** is working well for it's use, and you don't want it to get skewed by new data.

**What techniques would be useful to train a neural network for a Robot that walks on wild/natural terrain that's uncertain?:**
- In terms of training the neural network, natural terrain is quite random in the sense that it's shape & arrangement is very unpredictable, example: It's very hard to model an "average" river, even if we got the average width, depth, speed of running water for lots of rivers, we'd still need to determine it's top-down shape, then simulate how another randomly generated element of the terrain is also generated.... This seems **unfeasible** 
- ^ This tells me that due to the terrain's highly-complex (seemingly random) pattern, I think **Online Learning** is a huge must for this task, this would improve the robot drastically since with more data, this complex problem now has more data-points so hopefully this allows us to anticipate the terrain to a higher degree. (more accurately)
- If we're going with Online Learning, the live data we're feeding to the training model is going to be unlabelled, since we can't label the data on the fly unfortunately. (therefore this task must be an *Online & Unsupervised learning task*) The neural network would still be able to look at other parameters such as Geo-location *(Maybe the model knows that the Terrain in Peru near [**Machu Picchu**](https://www.peruforless.com/blog/machu-picchu-mountain/) is very Mountainous, Rocky & Ridged)*
- I think the last technique i'd apply for this task is to also use some kind of metric of success & apply a re-enforcement task to help the task to evolve in the right direction. This metric could be a measure of how long it takes the average human to preform the walk, versus the robot `Human Travel Time / Robot Travel Time` for example. from here I'd know the bot preforms poorly if the human takes 10 minutes and the robot takes 20. (10/20) = ~50% Success

**TL:DR ->** Online & Unsupervised Learning Task, that tries to optimize for success, and learns from it's successes.
 

**What is out-of-core Learning?:**




