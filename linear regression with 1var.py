import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_pd = pd.read_csv('ex1data1.txt', names=['pop', 'profit'])  # read csv file and return a dataframe obj
x = data_pd['pop']
y = data_pd['profit']
# plot the data
plt.scatter(x, y, marker='x')
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.show()


# implement of cost function
arr1 = np.array(list(x)).reshape((97, 1))
X = np.insert(arr1, 0, 1, axis=1)
#X = (X - X.mean())/(X.max() - X.min())
#print(X.min(), X.max())
Y = np.array(list(y)).reshape((97, 1))
THETA_INI = np.zeros((2, 1))
m = X.shape[0]
print(m)
print(X.shape, Y.shape, THETA_INI.shape)
def cost_function(THETA_INI, X, Y):
    temp = np.power((np.dot(X, THETA_INI) - Y), 2)
    J = np.sum(temp) / (2 * X.shape[0])
    return J
print(cost_function(THETA_INI, X, Y))


# implement of gradient descent
alpha = 0.01
epoch = 1500
def gradient_descent(THETA_INI, X, Y):
    THETA_tempt = THETA_INI
    cost_list = []
    for i in range(epoch):
        #print('the number of cost: ' + str(i), cost_list)
        list1 = THETA_tempt.flatten('F')
        for j in range(2):
            list1[j] = list1[j] - (alpha/m) * np.sum((np.dot(X, THETA_tempt) - Y) * np.array(X[:, j]).reshape((97, 1)))
        THETA_tempt = np.array(list1).reshape((2, 1))
        cost = cost_function(THETA_tempt, X, Y)
        cost_list.append(cost)
    return cost_list, THETA_tempt
cost_list, THETA_final = gradient_descent(THETA_INI, X, Y)
print(THETA_final)
'''
  为了能够同时更新模型参数，利用一个嵌套循环。外层是需要迭代的次数，把每一次更新后的theta代入记录代价函数值。 
  内层对模型参数进行更新。把模型参数从ndarray转成list形式。然后运用梯度下降算法更新参数。

'''


# plot y:cost_function, x:the number of iteration
def plot_testfun(cost_list):
    plt.figure(figsize=(10, 6))         # if I dont add this line, the figure will not be correct.why?
    plt.plot(list(range(1500)), cost_list)
    plt.xlabel('the num of iteration')
    plt.ylabel('cost_function')
    plt.show()
plot_testfun(cost_list)

# plot the final h(x)
def plot_h():
    plt.figure(figsize=(10, 6))
    plt.scatter(data_pd['pop'], data_pd['profit'], marker='x', label='training sets')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    x = np.linspace(5, 25)
    y = THETA_final[0] + THETA_final[1] * x
    plt.plot(x, y, 'm', label='hypothesis h(x)=  h(x) = %0.2f + %0.2fx'%(THETA_final[0],THETA_final[1]))
    plt.grid()
    plt.legend()
    plt.show()
plot_h()

