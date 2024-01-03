#! SOURCE I'm copying:
# https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np

# to make this program's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== Building an Image Classifier =====

from tensorflow import keras
print("\nTF Version: "+tf.__version__)

# Load in Fashion MNIST, The Load_Data function automatically Returns TRAINING & TEST data!
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# 60K training entries & 10K testing entries | of 28 x 28 pixel images
print(f"Fashion MNIST\n Training set {X_train_full.shape} \n Testing set {X_test.shape}")

#* Pixels of the imgs are 0-255 values, so we're normalizing to the 0-1 range below:

# Split into Training and Validation (testing) sets & Normalizing
X_valid, X_train = X_train_full[:5000] / 255 , X_train_full[5000:] / 255
X_test = X_test / 255

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# Y Labels for y_valid, y_train and y_test (Indexes to Str labels)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


#* ======= Show off the dataset with labels ========

# n_rows = 4
# n_cols = 10
# plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
# for row in range(n_rows):
#     for col in range(n_cols):
#         index = n_cols * row + col
#         plt.subplot(n_rows, n_cols, index + 1)
#         plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
#         plt.axis('off')
#         plt.title(class_names[y_train[index]], fontsize=12)
# plt.subplots_adjust(wspace=0.2, hspace=0.5)
# plt.show()


def build_and_train_classification_model(): #* ======= Building NN Architecture with Keras Sequential API ======

    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28])) #As our inputs are 28x28 px imgs
    model.add(keras.layers.Dense(300, activation=keras.activations.relu)) #! equiv to activation="relu" 
    model.add(keras.layers.Dense(100, activation=keras.activations.relu))
    model.add(keras.layers.Dense(10, activation="softmax")) #Softmax is used here for probablistic activation (good for classification)

    #Very nice printing for robust view of Model Layers & Arch
    # print("\n\n", model.summary())

    hidden1 = model.layers[1]
    weights, biases = hidden1.get_weights()

    print("DENSE LAYER 300 WEIGHTS : ", weights) #Randomly initialized weights, so that backprop will be differ
    #? Biases start off as all Zeros, and get updated upon training

    #* using Pydot to draw/save the img of the model arch.
    # keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)

    model.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])

    #* LOSS: We've got Sparse Labels which are mutually exclusive (Each img is 1 label, it can't be a Hoodie and Sandals at once)
    #* Cross Entropy Loss is ideal for Classification Tasks

    #* OPTIMIZER: SGD is fine, it's just adjusting the weights over and over again, to minimize loss

    #* METRICS: Since it's a classifier, we need accuracy, so it best matches labels to images (classified accurately!!)


    #* ========== Training of Model ==========
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid))


    #Evaluate model on test data (data & labels)
    model.evaluate(X_test, y_test)

    # Save trained model
    #! Warning, .h5 is legacy, use .keras
    model.save("my_fashion_mnist_model.h5") 


def loading_and_predicting():
    model = keras.models.load_model("my_fashion_mnist_model.h5")

    # Using the Test Set to make predictions as we don't got any other data

    X_new = X_test[:3] # Take first 3 entries from X_test

    def predict_fashion_img(X_entry):

        # This will change shape from (28, 28) to (1, 28, 28) as predict expects a batch
        X_entry = np.expand_dims(X_entry, axis=0)  
        
        # Make predictions
        y_proba = model.predict(X_entry)
        y_proba.round(2)

        # [0.00, 0.00 , ... 0.99 ]  -> IDX: 9 for EX:
        classInt = np.argmax(y_proba, axis=1)

        Label_prediction = class_names[classInt[0]]
        
        # Plotting
        plt.imshow(X_entry[0], cmap="binary")  # Adjust the colormap if needed
        plt.title(f"Predicted: {Label_prediction}")
        plt.axis('off')  # Turn off axis numbers and labels
        plt.show()

    predict_fashion_img(X_new[0])
    predict_fashion_img(X_new[1])
    predict_fashion_img(X_new[2])

#build_and_train_classification_model()
loading_and_predicting()

