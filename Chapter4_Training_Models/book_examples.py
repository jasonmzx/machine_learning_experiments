
# Numbers & Plotting libs
import numpy as np
import matplotlib.pyplot as plt 

# ML Stuff
import sklearn
assert sklearn.__version__ >= "0.20"

#Generation of Linear points with gaussian noise (Sample X-Y data for linear regression testing)
#Let's make our true linear equation be: 
#& y = 5x + 2

#TODO: how does np.random.rand( ? , ? ) work?

noise_coefficient = 1.25

# np.random.rand( d0, d1) -> this constructor will initialize a 2D shape of height d0 and object dim of d1

X = 2 * np.random.rand(100,1) #100 entries of a 1D object, (List of Integers lol)

Y = 5 * X + 2 + noise_coefficient * np.random.rand(100,1) # The last part adds some noise to the data, this simulates real world data 

#Showing of plot randomly generated items (Linear points with noise)

plt.plot(X, Y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
#save_fig("generated_data_plot")
plt.show()


#Preforming Scikit's built-in LinearRegression function
#   This regression model calculates the least squares residuals thing just like in stats

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
#lin_reg.intercept_, lin_reg.coef_ <- Gives you intercept (B) and Slope (M)


#Using the some "Normal Equation" 
    #In linear regression analysis, the normal equations are a 
    # system of equations whose solution is the Ordinary Least Squares 
    # (OLS) estimator of the regression coefficients. 

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

print(theta_best) #This result is identical to what's produced with LinearRegression fit 



# Let's say we don't know about any of these regressors, let's use Gradient Descent instead!
#! Known as Batch Gradient Descent

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
