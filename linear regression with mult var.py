import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# obtain data and apply scaling
data_pd = pd.read_csv('ex1data2.txt', names=['size', 'bedrooms', 'price'])
#print(data_pd.head())
x1 = data_pd['size']
x2 = data_pd['bedrooms']
y = data_pd['price']
m = len(x1)     #m = 47
X1 = np.array(x1.values).reshape((47, 1))
X2 = np.array(x2.values).reshape((47, 1))
X1 = (X1 - np.mean(X1)) / (np.max(X1) - np.min(X1))
X2 = (X2 - np.mean(X2)) / (np.max(X2) - np.min(X1))
X = np.concatenate((np.ones((47, 1)), X1, X2), axis=1)    # X (47, 3)
Y  = np.array(y.values).reshape((47, 1))


# implement of cost function
THETA_INI  = np.zeros((3, 1))
print(X.shape, THETA_INI.shape, Y.shape)
def cost_func(X, Y, THETA_INI):
    tempt = np.dot(X, THETA_INI) - Y
    tempt = np.power(tempt, 2)
    J = np.sum(tempt) / 2 * X.shape[0]
    return  J


# implement of gradient descent
alpha = 0.01
epoch = 1500
def gradient_descent(X, Y, THETA_INI):
    cost_list = [ ]
    THETA_tempt = THETA_INI
    for i in range(1500):
        list1 = THETA_tempt.flatten('F')
        for j in range(2):
            list1[j] = list1[j] - (alpha / m) * np.sum((np.dot(X, THETA_tempt) - Y) * np.array(X[:, j]).reshape((47, 1)))
        THETA_tempt = np.array(list1).reshape((3, 1))
        cost = cost_func(X, Y, THETA_tempt)
        cost_list.append(cost)
    return cost_list, THETA_tempt
cost_list, THETA_final_1 = gradient_descent(X, Y, THETA_INI)
print(THETA_final_1)


# implement of  normal equation
def normal_equation(X, Y):
    tempt = np.dot(X.T, X)
    tempt = np.linalg.inv(tempt)
    tempt = np.dot(tempt, X.T)
    THETA = np.dot(tempt, Y)
    return THETA
THETA_final_2 = normal_equation(X, Y)
print(THETA_final_2)


#plot
def plt_figure():
    plt.figure(figsize=(10, 6))
    plt.plot(range(1500), cost_list, 'r')
    plt.xlabel('the number of itineration')
    plt.ylabel('cost_function')
    plt.grid()
    plt.show()
plt_figure()


