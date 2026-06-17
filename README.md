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

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Loading the Data
data = load_diabetes()
X = data.data         
Y = data.target       

# 2. Preparing the data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X = feature_scaler.fit_transform(X)
y = Y.reshape(-1, 1)
y = target_scaler.fit_transform(y)
y = y.ravel()

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# 3. Initializing Parameters
m, n = X.shape   
w = np.zeros(n)  
b = 0            

# 4. Defining Core Functions
def predict(X, w, b):
    return np.dot(X, w) + b

def compute_cost(X, y, w, b):
    m = len(y)
    y_pred = predict(X, w, b)
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
    return cost

def compute_gradients(X, y, w, b):
    m = len(y)
    y_pred = predict(X, w, b)
    error = y_pred - y
    dw = (1 / m) * np.dot(X.T, error)
    db = (1 / m) * np.sum(error)
    return dw, db

def update_parameters(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

# 5. Training the Model
learning_rate = 0.001
num_iterations = 5000
cost_history = []

for i in range(num_iterations):
    y_pred = predict(X, w, b)
    cost = compute_cost(X, y, w, b)
    dw, db = compute_gradients(X, y, w, b)
    w, b = update_parameters(w, b, dw, db, learning_rate)

    val_cost = compute_cost(X_val, y_val, w, b)

    if i % 500 == 0:
      cost_history.append([cost, val_cost])
      print(f'Iteration {i}: Cost = {cost:.4f}: Val Cost = {val_cost:.4f}')

# 6. Saving parameters
param = {'weights': w.tolist(), 'bias': b}
with open('param.json', 'w') as f:
    json.dump(param, f)

# 7. Evaluating Model Performance with Test Data
y_pred_test = predict(X_test, w, b)

mae = np.abs(y_test - y_pred_test).mean()
mse = ((y_test - y_pred_test)**2).mean()
rmse = np.sqrt(((y_test - y_pred_test)**2).sum()/len(y_test))

SS_res = np.sum((y_test - y_pred_test)**2)        
SS_tot = np.sum((y_test - np.mean(y_test))**2)    
r2_ = 1 - (SS_res / SS_tot)

# 8. Comparison with Sklearn Linear Regression
model = SGDRegressor(
    loss='squared_error',
    alpha=0.0,
    learning_rate='constant',  
    eta0=0.01,                 
    max_iter=1000,
    random_state=42
)
model.fit(X, y)  
y_pred_test_sk = model.predict(X_test)

mse_sk = mean_squared_error(y_test, y_pred_test_sk)
r2_sk = r2_score(y_test, y_pred_test_sk)

print("\n--- FINAL METRICS ---")
print(f"Scratch MAE: {mae:.4f}")
print(f"Scratch MSE: {mse:.4f}")
print(f"Scratch RMSE: {rmse:.4f}")
print(f"Scratch R2: {r2_:.4f}")
print(f"Sklearn MSE: {mse_sk:.4f}")
print(f"Sklearn R2: {r2_sk:.4f}")
