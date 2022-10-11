# -*- coding: utf-8 -*-
"""cs7cs4.ipynb

# Q1

## a
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
# %matplotlib inline

df = pd.read_csv('week3.txt',comment="#", header=None)
df

X = df.iloc[:,:2]
X

y = df.iloc[:,2]
y

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10)) 
ax = fig.add_subplot(111, projection ='3d') 
ax.scatter(X[0], X[1] , y)
ax.set(title='Training data', xlabel='X1', ylabel='X2', zlabel='Y')
plt.show()

"""## b"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
poly = PolynomialFeatures(degree=5)
X5poly = poly.fit_transform(X)

for f in poly.get_feature_names_out():
  print(f)

C_range = [1, 10, 1000, 10000]
for C in C_range:
    model = Lasso(alpha=1/(2*C)).fit(X5poly, y)
    theta = model.coef_
    theta0 = model.intercept_
    y_pred = model.predict(X5poly)
    j = mean_squared_error(y, y_pred)
    print(f"C = {C}\nmse = {j:.4f}\n")
    print(f'theta = {theta}\ntheta0 = {theta0}\n')

"""## c"""

Xtest = []
grid = np.linspace(-5,5)
for i in grid:
    for j in grid:
        Xtest.append([i,j])

Xtest = np.array(Xtest)
Xtest = PolynomialFeatures(5).fit_transform(Xtest)

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

C_range = [1, 10, 1000, 10000]
for C in C_range:
    model = Lasso(alpha=1/(2*C))
    model.fit(X5poly, y)
    Z = model.predict(Xtest)
    fig = plt.figure(figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0], X[1], y, c='purple', label="Training data")
    surf = ax.plot_trisurf(Xtest[:,1], Xtest[:,2], Z, 
                           cmap=cm.jet, alpha=0.8, linewidth=0, antialiased=True)
    ax.set_title(f'Lasso prediction surface with C = {C}')
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set(xlabel='X1', ylabel='X2', zlabel='Y')
    ax.legend(loc='lower right')
    plt.show()

"""## e"""

# (i)(e)
from sklearn.linear_model import Ridge

C_range = [1e-7, 1e-5, 1e-1, 1]
for C in C_range:
    model = Ridge(alpha=1/(2*C)).fit(X5poly, y)
    theta = model.coef_
    theta0 = model.intercept_
    y_pred = model.predict(X5poly)
    j = mean_squared_error(y, y_pred)
    print(f"C = {C}\nmse = {j:.4f}\n")
    print(f'theta = {theta}\ntheta0 = {theta0}\n')
    
    model.fit(X5poly, y)
    Z = model.predict(Xtest)
    fig = plt.figure(figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0], X[1], y, c='purple', label="Training data")
    surf = ax.plot_trisurf(Xtest[:,1], Xtest[:,2], Z, 
                           cmap=cm.jet, alpha=0.8, linewidth=0, antialiased=True)
    ax.set_title(f'Lasso prediction surface with C = {C}')
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set(xlabel='X1', ylabel='X2', zlabel='Y')
    ax.legend(loc='lower right')
    plt.show()

"""# Q2

## a
"""

from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

test_std = []
test_mean = []
train_std = []
train_mean = []

plt.figure(figsize=(8, 6), dpi=120)

C_range = [1, 3, 5, 7, 10, 30, 50, 70, 100, 300, 500]
kfcv = KFold(n_splits=5)

for C in C_range:
    model = Lasso(alpha=1/(2*C))
    
    test_mse = []
    train_mse = []
    for train, test in kfcv.split(X5poly):
        model.fit(X5poly[train], y[train])
        y_pred_test = model.predict(X5poly[test])
        y_pred_train = model.predict(X5poly[train])
        
        test_mse.append(mean_squared_error(y[test], y_pred_test))
        train_mse.append(mean_squared_error(y[train], y_pred_train))

    test_mean .append(np.mean(test_mse))
    test_std.append(np.std(test_mse))
    train_mean.append(np.mean(train_mse))
    train_std.append(np.std(train_mse))

plt.errorbar(C_range, test_mean, yerr=test_std, linewidth=3, label="Test data")
plt.errorbar(C_range, train_mean, yerr=train_std, linewidth=3, label="Training data")
plt.title("mean and standard deviation of the prediction error vs C (Lasso)")
plt.xlabel('C')
plt.ylabel("Mean square error")
plt.legend()
plt.show()

"""## c"""

test_std = []
test_mean = []
train_std = []
train_mean = []

plt.figure(figsize=(8, 6), dpi=120)

C_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
kfcv = KFold(n_splits=5)

for C in C_range:
    model = Ridge(alpha=1/(2*C))
    
    test_mse = []
    train_mse = []
    for train, test in kfcv.split(X5poly):
        model.fit(X5poly[train], y[train])
        y_pred_test = model.predict(X5poly[test])
        y_pred_train = model.predict(X5poly[train])
        
        test_mse.append(mean_squared_error(y[test], y_pred_test))
        train_mse.append(mean_squared_error(y[train], y_pred_train))

    test_mean .append(np.mean(test_mse))
    test_std.append(np.std(test_mse))
    train_mean.append(np.mean(train_mse))
    train_std.append(np.std(train_mse))

plt.errorbar(C_range, test_mean, yerr=test_std, linewidth=3, label="Test data")
plt.errorbar(C_range, train_mean, yerr=train_std, linewidth=3, label="Training data")
plt.title("mean and standard deviation of the prediction error vs C (Lasso)")
plt.xlabel('C')
plt.ylabel("Mean square error")
plt.legend()
plt.show()

