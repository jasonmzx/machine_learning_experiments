## Cross Entropy

https://machinelearningmastery.com/cross-entropy-for-machine-learning/

**Information** : N. bits to encode & transmit an event *(How much information needed for encoding & transmission of this event?)*

**Low Probability Event** *(surprising)*: More information.
**Higher Probability Event** *(unsurprising)*: Less information.

Information, h(x) given an event x can be calculated with Probability P(x). <br>
`h(x) = -log(P(x))`

Certainly! Your notes on Information are a great starting point. Let's extend this to create similar notes for Entropy and Cross-Entropy.

---

### **Entropy (H)**

**Definition**: Entropy is a measure of the average amount of information needed to encode and transmit events from a probability distribution. It quantifies the uncertainty or randomness of a probability distribution.

**Interpretation**:
- **Higher Entropy**: If the distribution is uniform *(all events have equal probability)*, the entropy is maximized. This represents maximum uncertainty or disorder. **EX:** *Dice Roll, all numbers are equality as likely, so Entropy high asf*

- **Lower Entropy**: If some events are much more likely than others, the entropy is lower. This represents less uncertainty.

**Use in Information Theory**:
- Entropy is used as a baseline measure of the inherent unpredictability in a dataset or a signal.

---

### **Cross-Entropy (H')**

**Definition**: Cross-Entropy is a measure of the difference between *two probability distributions.* It quantifies how much information is needed to represent or encode events from one distribution using the probability distribution of another distribution.

**Interpretation**:
- **Low Cross-Entropy**: Indicates that the two distributions are similar. Less additional information is needed to encode events from one distribution using the other.
- **High Cross-Entropy**: Indicates a greater difference between the distributions. More additional information is needed for encoding.

---

### Why Cross-Entropy for Classification Tasks? 

As Cross-Entropy basically looks at the different between 2 probablistic distributions, but more specifically, it quantifies how a predicted probability distribution diverges from the actual distribution. It's a measure of surprise, originating from information theory! 
<br>
It's very useful for Classification where we see the Output Array *(The 10 ouputs in the softmax layer for ex. )* with varying confidence (probabilities) and we can compare this directly to like a One-Hot Encoded Distribution where all probabilites are 0, except for the correct category, in which the probability is 100%. This is better than traditional error loss (like MSE). 
<br>
Cross-Entropy can actually take into account confidence of a vote *(demonstrated as a high probability guess on the model's part)* as well as well as directly penalizes the logarithm of the probability assigned to the correct class, which inherently aligns with the goal of classification to maximize the probability of the correct class. MSE, on the other hand, doesn't directly align with this goal and can lead to slower convergence in training due to its different mathematical properties.

---

https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb

