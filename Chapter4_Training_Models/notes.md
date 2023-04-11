
| Algorithm | Large m | Out-of-core support | Large-n | Hyperparameters | Scaling Required | Scikit-learn built-in callback |
|-----------|---------|---------------------|---------|-----------------|------------------|--------------------------------|
| Normal Equation | Not suitable | No | Suitable | None | No | N/A |
| SVD | Suitable | No | Suitable | None | No | LinearRegression |
| Batch Gradient Descent | Not suitable | No | Suitable | Learning rate, number of iterations | Yes | SGDRegressor |
| Stochastic Gradient Descent | Suitable | Yes | Suitable | Learning rate, number of iterations | Yes | SGDRegressor |
| Mini-batch Descent | Suitable | Yes | Suitable | Learning rate, number of iterations, batch size | Yes | SGDRegressor |

**Note:** In the "Scaling Required" column, "Yes" means that the algorithm requires feature scaling (i.e., scaling the input data) to perform well. The "Scikit-learn built-in callback" column refers to whether the scikit-learn library provides a built-in function to track and log the optimization process of the algorithm during training.

*Credits to ChatGPT for that one ^^^*