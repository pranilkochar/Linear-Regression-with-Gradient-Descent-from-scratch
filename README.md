# Linear Regression with Gradient Descent (From Scratch)

This repository contains a complete, end-to-end implementation of **Linear Regression using Gradient Descent** built entirely from scratch in Python. The custom implementation handles the mathematical computations manually to demonstrate the inner workings of gradient-based optimization. It is subsequently benchmarked against Scikit-Learn's robust `SGDRegressor`.

---

## 🧠 Introduction

Linear Regression aims to model the relationship between independent variables (features) and a continuous dependent variable (target) by fitting a linear equation to the observed data. The goal is to find the optimal weights and bias that minimize the difference between predicted and actual values.

**Gradient Descent** is an iterative optimization algorithm used to minimize the cost function. It works by calculating the gradient (the slope) of the cost function with respect to each parameter, and taking small steps in the opposite direction of the gradient until the model converges to the global minimum.

---

## 📊 Mathematical Foundations

This Python script natively implements the following mathematical formulas:

### 1. Prediction Function
The linear hypothesis used to predict the target variable is:

$$\hat{y}_{(i)} = \sum_{j=1}^{n} w_j x_j^{(i)} + b$$

### 2. Cost Function (Mean Squared Error)
To measure how well the model is performing, the Cost Function evaluates the average squared difference between predictions and actual values:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}_{(i)} - y_{(i)} \right)^2$$

### 3. Gradients Calculation
To update the parameters, partial derivatives of the cost function are calculated with respect to weights and bias:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_{(i)} - y_{(i)} \right) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_{(i)} - y_{(i)} \right)$$

### 4. Parameter Update Rule
The parameters are shifted iteratively in the direction of steepest descent by scaling the gradients with the learning rate $\alpha$:

$$w := w - \alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

---

## 💻 Python Source Code

### Step 1 & 2: The Manual Approach
First, we will import libraries & load dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes  #loading diabetic dataset from sklearn datasets library
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()

X = data.data         # Feature matrix (shape: [442, 10])
Y = data.target       # Target vector (shape: [442,])

print('Feature names:', data.feature_names)
print('Feature values:', X[:1])
print('First five target values:', Y[:5])
print('x shape:' ,X.shape)
print('y shape:' ,Y.shape)

```
**Expected Output:**
```text
Feature names: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Feature values: [[ 0.03807591  0.05068012  0.06169621  0.02187239 -0.0442235  -0.03482076
  -0.04340085 -0.00259226  0.01990749 -0.01764613]]
First five target values: [151.  75. 141. 206. 135.]
x shape: (442, 10)
y shape: (442,)
```

### Step 3: Data Preparation
The following code standardizes the features and splits the dataset into training, validation & testing sets:

StandardScaler: Standardizes the feature matrix by removing the mean and scaling to unit variance.
train_test_valid_split: Splits the dataset into training (80%) and testing (10%) & validation (10%) sets.
The feature matrix X is transformed, and the dataset is divided into X, X_test, y, and y_test.

```python
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X = feature_scaler.fit_transform(X)

y = y.reshape(-1, 1)
y = target_scaler.fit_transform(y)
y = y.ravel()

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print("Train Size : ",X.shape) #80%
print("Test Size : ",X_test.shape) #10%
print("Validation Size : ",X_val.shape) #10%

```
**Expected Output:**
```text
Train Size :  (353, 10)
Test Size :  (45, 10)
Validation Size :  (44, 10)
```


### Step 4: Initializing Parameters
This code initializes the parameters for the linear regression model:

m, n: The shape of the feature matrix X, where m is the number of samples (442) and n is the number of features (10). w: The weight vector, initialized to zeros with shape [10,]. b: The bias term, initialized to 0.

```python
m, n = X.shape   # m = 353, n = 10
w = np.zeros(n)  # Weight vector (shape: [10,])
b = 0            # Bias term (scalar)
```


### Step 5: Defining the Prediction Function
The prediction function is given by:

$$
\large \hat{y}_{(i)} = \sum_{j=1}^{n} w_j x_j^{(i)} + b
$$



**Where:**

 $$ \hat{y}_{(i)} = \text{predicted target value for the } i\text{-th sample.} $$
 $$ w_j = \text{weight for the } j\text{-th feature.} $$
 $$ x_j^{(i)} = \text{value of the } j\text{-th feature for the } i\text{-th sample.} $$

 This code defines the **predict** function, which calculates the predicted target values for each sample using the feature matrix **X**, weight vector **w**, and bias term **b**:

- **X**: The feature matrix (shape: [m, n]), where **m** is the number of samples and **n** is the number of features.
- **w**: The weight vector (shape: [n,]), which contains the learned weights for each feature.
- **b**: The bias term (scalar), which is added to the weighted sum.

The function uses the following formula to compute the predictions:

$$
\hat{y} = Xw + b
$$

```python
def predict(X, w, b):
    return np.dot(X, w) + b
```


### Step 6: Defining the Cost Function
The cost function measures how well the model's predictions match the actual target values and is defined as:

$$
\large J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}_{(i)} - y_{(i)} \right)^2
$$

Where **m** is the number of training examples.


```python
def compute_cost(X, y, w, b):
    m = len(y)
    y_pred = predict(X, w, b)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

    return cost
```


### Step 7: Computing the Gradients
Gradients are computed to update the weights and bias during training. The partial derivatives of the cost function with respect to each weight and the bias are:

$$
\large \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_{(i)} - y_{(i)} \right) x_j^{(i)}
$$

.

$$
\large \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}_{(i)} - y_{(i)} \right)
$$


```python
def compute_gradients(X, y, w, b):

    m = len(y)
    y_pred = predict(X, w, b)
    error = y_pred - y

    dw = (1 / m) * np.dot(X.T, error)
    db = (1 / m) * np.sum(error)

    return dw, db
```



### Step 8: Updating Parameters Using Gradient Descent
Parameters are updated using the learning rate and the gradients calculated:

$$
\large w := w - \alpha \frac{\partial J}{\partial w}
$$

$$
\large b := b - \alpha \frac{\partial J}{\partial b}
$$

**Where:**

$$
\alpha = \text{learning rate}
$$


```python
def update_parameters(w, b, dw, db, learning_rate):

    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b
```


### Step 9: Training the model
This code performs the training of the linear regression model using gradient descent. It iteratively updates the model's parameters (weights and bias) and tracks the cost function:

- **learning_rate**: The step size for updating the parameters during each iteration. Set to **0.01**.
- **num_iterations**: The number of iterations for the gradient descent process. Set to **1000**.
- **cost_history**: A list that stores the cost value at each iteration to track the convergence of the model.

The training loop performs the following steps for each iteration:

1. **Prediction**: Uses the `predict` function to compute the predicted target values (**y_pred**).
2. **Cost Calculation**: Computes the cost (error) using the `compute_cost` function.
3. **Store Cost**: Appends the cost to the **cost_history** list for tracking.
4. **Gradient Calculation**: Computes the gradients for the weights and bias using `compute_gradients`.
5. **Parameter Update**: Updates the weights and bias using the `update_parameters` function with the calculated gradients and learning rate.
6. **Iteration Logging**: Every 100th iteration, prints the current iteration number and the corresponding cost.

```python
w = np.zeros(n)
b = 0

learning_rate = 0.001
num_iterations = 5000
cost_history = []

parameters = {}

for i in range(num_iterations):

    y_pred = predict(X, w, b)
    cost = compute_cost(X, y, w, b)
    dw, db = compute_gradients(X, y, w, b)
    w, b = update_parameters(w, b, dw, db, learning_rate)

    val_cost = compute_cost(X_val, y_val, w, b)

    if i % 500 == 0:
      cost_history.append([cost, val_cost])
      print(f'Iteration {i}: Cost = {cost:.4f}: Val Cost = {val_cost:.4f}')

      parameters = {'weights': w.tolist(), 'bias': b}

```
**Expected Output:**
```text
Iteration 0: Cost = 0.5126: Val Cost = 0.4155
Iteration 500: Cost = 0.2850: Val Cost = 0.2335
Iteration 1000: Cost = 0.2580: Val Cost = 0.2263
Iteration 1500: Cost = 0.2496: Val Cost = 0.2251
Iteration 2000: Cost = 0.2465: Val Cost = 0.2255
Iteration 2500: Cost = 0.2453: Val Cost = 0.2262
Iteration 3000: Cost = 0.2448: Val Cost = 0.2269
Iteration 3500: Cost = 0.2446: Val Cost = 0.2275
Iteration 4000: Cost = 0.2444: Val Cost = 0.2279
Iteration 4500: Cost = 0.2444: Val Cost = 0.2283
```

### Step 10: Saving the model's final weights & biases

```python
import json

param = {'weights': w.tolist(), 'bias': b}

with open('param.json', 'w') as f:
    json.dump(param, f) {'weights': w.tolist(), 'bias': b}

```

### Step 11: Evaluting model's performance
This code evaluates the performance of the trained linear regression model by calculating the final cost and Mean Squared Error (MSE), as well as printing the final learned parameters (weights and bias):

- **y_pred**: The predicted target values obtained from the `predict` function, using the final values of weights **w** and bias **b**.
- **final_cost**: The computed cost using the `compute_cost` function, representing the error between the predicted and actual values.
- **mse**: The Mean Squared Error, calculated as **2 times the final cost**, which is a common metric to evaluate regression models.


#### Key Metrics:

- **Residual Analysis**: The residuals are the differences between the actual and predicted values. This metric is calculated by taking the mean of the absolute differences between the actual test values (**y_test**) and predicted values (**y_pred_test**). A lower value indicates a better fit.
  
- **Mean Absolute Error (MAE)**: MAE is the average of the absolute differences between the actual and predicted values. It provides a simple measure of prediction accuracy.
  
- **Mean Squared Error (MSE)**: MSE is the average of the squared differences between the actual and predicted values. It penalizes larger errors more than MAE, making it sensitive to outliers.

- **Root Mean Squared Error (RMSE)**: RMSE is the square root of the MSE and provides an error metric in the same unit as the target variable, making it more interpretable.

- **R-squared (R²)**: R-squared measures the proportion of variance in the target variable that is predictable from the independent variables. A value closer to 1 indicates a better fit of the model to the data.

```python
y_pred_test = predict(X_test, w, b)

# Calculate metrics

print(f"Residual Analysis : {(np.abs(y_test - y_pred_test)).mean()}")

mae = np.abs(y_test - y_pred_test).mean()
mse = ((y_test - y_pred_test)**2).mean()
rmse = np.sqrt(((y_test - y_pred_test)**2).sum()/len(y_test))


print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


SS_res = np.sum((y_test - y_pred_test)**2)        # Sum of squares of residuals(diff bet actual and predicted values)
SS_tot = np.sum((y_test - np.mean(y_test))**2)    # Total sum of squares

r2_ = 1 - (SS_res / SS_tot)

print(f"R-squared: {r2_:.4f}")

```
**Expected Output:**
```text
Residual Analysis : 0.5781057679914285
Mean Absolute Error (MAE): 0.5781
Mean Squared Error (MSE): 0.5140
Root Mean Squared Error (RMSE): 0.7169
R-squared: 0.4664
```


### Step 12: Conclusion

In this project, we successfully implemented a simple linear regression model from scratch using gradient descent. The following key steps were involved:

1. **Data Preparation**: We loaded and explored the dataset, performing necessary preprocessing, such as scaling features and splitting the data into training and test sets.
2. **Model Initialization**: The model's parameters (weights and bias) were initialized to zero.
3. **Prediction and Cost Calculation**: We defined a prediction function and a cost function to evaluate the accuracy of the model.
4. **Gradient Descent**: We calculated the gradients and applied gradient descent to update the model parameters iteratively.
5. **Model Training**: Over 1000 iterations, the model was trained, and the weights and bias were updated.
6. **Model Evaluation**: Finally, we evaluated the model's performance using cost and mean squared error (MSE), and printed the final learned parameters.

### Key Findings:
- The linear regression model was successfully trained and optimized using gradient descent.
- The final learned weights and bias were printed, showing the results of the optimization.
- The model's performance was assessed using the cost and MSE, providing a clear understanding of its predictive ability.

### Benefits of Linear Regression:
- **Simplicity and Interpretability**: Linear regression is easy to implement and understand, making it a great starting point for many machine learning problems.
- **Efficiency**: The model is computationally inexpensive and can handle large datasets effectively.
- **Well-Suited for Linearly-Related Data**: It is ideal for predicting continuous target variables that have a linear relationship with the features.

### Drawbacks of Linear Regression:
- **Assumption of Linearity**: Linear regression assumes a linear relationship between the features and target variable, which may not always hold true.
- **Sensitivity to Outliers**: Outliers can have a significant impact on the model's performance, as linear regression is sensitive to extreme values.
- **Multicollinearity**: High correlation between input features can lead to unreliable parameter estimates and degrade the model's performance.

In conclusion, while linear regression provides an effective and interpretable model for problems with a linear relationship, its limitations in handling non-linearity and outliers must be considered. For more complex relationships, other models such as polynomial regression or more advanced machine learning algorithms may be necessary.


### Step 13: Building same model using ready to use SK Learn libraries & comparing it with our model which we build from scratch
In this section, we use **scikit-learn** to train a linear regression model and evaluate its performance using the test data. The model is trained on the features (**X**) and target values (**y**) and then evaluated using **Mean Squared Error (MSE)** and **R-squared (R²)** metrics.

### Code Explanation:

1. **Model Initialization**:
   - `model = LinearRegression()`: Initializes a linear regression model using **scikit-learn**'s `LinearRegression` class.

2. **Model Training**:
   - `model.fit(X, y)`: Fits the model to the training data (**X**, **y**) by finding the optimal values for the weights and bias.

3. **Prediction**:
   - `y_pred_test = model.predict(X_test)`: Uses the trained model to predict the target values (**y_pred_test**) for the test data (**X_test**).

4. **Model Evaluation**:
   - `mse = mean_squared_error(y_test, y_pred_test)`: Computes the **Mean Squared Error (MSE)** between the actual target values (**y_test**) and the predicted values (**y_pred_test**). MSE measures the average squared difference between the predicted and actual values.
   - `r2 = r2_score(y_test, y_pred_test)`: Calculates the **R-squared (R²)** score, which indicates the proportion of the variance in the target variable that is explained by the model.

5. **Print Results**:
   - The code prints the computed **MSE** and **R-squared** values, providing an understanding of how well the model fits the data.

```python
model = SGDRegressor(
    loss='squared_error',
    alpha=0.0,
    learning_rate='constant',  # Set to 'constant' for a fixed learning rate
    eta0=0.01,                 # Initial learning rate (will remain fixed)
    max_iter=1000,
    random_state=42
    # tol = None                    # no early stopping
  )


model.fit(X, y)  # Use training data for fitting

# 4. Make predictions on the test set
y_pred_test = model.predict(X_test)

# 5. Evaluate the model
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

print('sklearn model weights : ' , model.coef_)
print('Scratch code we created model weights : ' , w)

```
**Expected Output:**
```text
Mean Squared Error: 0.5233
R-squared: 0.4567

sklearn model weights :  [ 0.06237818 -0.15471557  0.3444035   0.21401584 -0.23088024  0.00364137
 -0.08773289  0.13197388  0.33041899  0.05962125]
Scratch code we created model weights :  [ 0.02485687 -0.14711889  0.34126827  0.21068682 -0.06286226 -0.07740788
 -0.13253158  0.09256726  0.25598774  0.04181697]
```

