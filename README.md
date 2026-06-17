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

