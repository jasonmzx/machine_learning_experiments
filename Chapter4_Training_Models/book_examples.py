
# Numbers & Plotting libs
import numpy as np
import matplotlib.pyplot as plt 
import numpy.random as rnd

# ML Stuff
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.linear_model import LinearRegression

#Generation of Linear points with gaussian noise (Sample X-Y data for linear regression testing)
#Let's make our true linear equation be: 
#& y = 5x + 2

#TODO: how does np.random.rand( ? , ? ) work?

#Global Data Declarations

#! LINEAR DATA GEN [ + NOISE ]

noise_coefficient = 1.25

# np.random.rand( d0, d1) -> this constructor will initialize a 2D shape of height d0 and object dim of d1

X = 2 * np.random.rand(100,1) #100 entries of a 1D object, (List of Integers lol)

Y = 5 * X + 2 + noise_coefficient * np.random.rand(100,1) # The last part adds some noise to the data, this simulates real world data 

#Generating and showing of plot linear placed items with a bit of random noise on the items
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
 
 #! QUADRACTIC DATA GEN [ + NOISE ]

np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3 #Assign at random X points, kinda simulates data clusters n stuff
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1) # X^2 + X + 2 , + noise (100 to 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()


print("Dataset(s) generated...")


def normal_and_gd_model_fitting():


    #Preforming Scikit's built-in LinearRegression function
    #   This regression model calculates the least squares residuals thing just like in stats

    lin_reg = LinearRegression()
    lin_reg.fit(X, Y)
    #lin_reg.intercept_, lin_reg.coef_ <- Gives you intercept (B) and Slope (M)


    #Using the some "Normal Equation" 
        #In linear regression analysis, the normal equations are a 
        # system of equations whose solution is the Ordinary Least Squares 
        # (OLS) estimator of the regression coefficients. 

        # X^T . X . X^T . Y

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

    print(theta_best) #This result is identical to what's produced with LinearRegression fit 

    # Let's say we don't know about any of these regressors, let's use Gradient Descent instead!
    #& Known as Batch Gradient Descent

    eta = 0.1  # learning rate
    n_iterations = 1000
    m = 100

    theta = np.random.randn(2,1)  # random initialization

    for iteration in range(n_iterations):

        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - Y) #This is the Gradient Vector of the Cost Function

        theta = theta - eta * gradients #Reassignment theta with an improved gradient (Amplified by learning rate)

    print("\n Gradient Descent Fit: ")
    print(theta)



    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  
    X_new_b.dot(theta)


    # Some visualizations of this:
    theta_path_bgd = []

    #This fn essentially just does BGD and plots every iteration as it's moving close to a minima

    def plot_gradient_descent(theta, eta, theta_path=None):
        m = len(X_b)
        plt.plot(X, Y, "b.")
        n_iterations = 1000
        for iteration in range(n_iterations):
            if iteration < 10:
                y_predict = X_new_b.dot(theta)
                style = "b-" if iteration > 0 else "r--"
                plt.plot(X_new, y_predict, style)
            gradients = 2/m * X_b.T.dot(X_b.dot(theta) - Y)
            theta = theta - eta * gradients
            if theta_path is not None:
                theta_path.append(theta)
        plt.xlabel("$x_1$", fontsize=18) #Some cool subscripting you can do in matplotlib
        plt.axis([0, 2, 0, 15])
        plt.title(r"$\eta = {}$".format(eta), fontsize=16)

    np.random.seed(42)
    theta = np.random.randn(2,1)  # random initialization

    plt.figure(figsize=(10,4))
    plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(133); plot_gradient_descent(theta, eta=0.5)

    plt.show()


#SGD takes 1 instance and optimizes based on only that 1 moving arround
#SGD "bounces arround" so to speak, since it's Schocastic (random) however it always tends to a local minimum (Lowest cost function)
#SGD is good when the cost function is irregular, as Batch GD, would get stuck at a local minima
#Whereas SGD could eventually bounce out of there to the global minima, which optimizes the cost func. much better

def sgd_model_fitting():
    epochs = 50
    t0, t1 = 5, 50 #learing schedule hyper-parameters
    m = 100


    def learning_schedule(t):
        return t0 / (t+t1)
    
    # 2 x 1 rngs
    theta = np.random.randn(2,1)

    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = Y[random_index:random_index+1]

            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + 1)
            theta = theta - eta * gradients

    print(theta)
    return

#TODO: Implement Mini-Batch fn
#TODO: Implement Polynomial regression fns


#* Main Function
def m():

    print("Input method to execute... gd , sgd")
    inp = input()

    match inp:
        case "gd": # Linear Model fitting Using Normal Equation, and Gradient Descent
            print("You've chosen, normal equation & gd: ")
            return normal_and_gd_model_fitting()
        case "sgd": #Linear Model fitting using Sochastic Gradient Descent
            print("You've chosen, normal sgd: ")
            return sgd_model_fitting() 

m()