"""
super simple neural networks(using numpy,snn.py not using numpy)
  * only one neuron in only the one hidden layer
  * input x is scalar (0-dimension)
  * using logistic function as the activation function

input layer:
    x: scalar
parameters:
    w: scalar
    b: scalar
output:
    y \in [0,1] or  p \in {0,1}
structure:

   x ->   w*x + b   ->   logistic function  -> output
        -----------      -----------------
             |                    |
             V                    V
         one neuron     activation function

about it:
    it's a simple project for human learning how machine learning
    by orczhou.com
"""
import numpy as np
import math

# function_f:
# x   : scalar or ndarray
# w   : scalar
# b   : scalar
def function_f(x,w,b):
    return 1/(1+np.exp(-(x*w+b)))

# initial w,b
w,b = (0,0)

# samples
x_train = np.array([0,1,2,3,4,5])
y_train = np.array([0,0,0,0,0,1])
#y_train = np.array([0,0,0,1,1,1])

# m for sample counts
m = x_train.shape[0]

iteration_count = 50000
learning_rate   = 0.01
cost_reduce_threshold = 0.000001

# Gradient caculate
# w_p: current w
# b_p: current b
def gradient_caculate(w_p,b_p):
    gradient_w = np.sum((function_f(x_train,w_p,b_p) - y_train)*x_train)
    gradient_b = np.sum(function_f(x_train,w_p,b_p) - y_train)
    return gradient_w,gradient_b

def cost_function(w_p,b_p,x_p,y_p):
    hat_y = function_f(x_p,w_p,b_p)
    c = np.sum(-y_p*np.log(hat_y) - (1-y_p)*np.log(1-hat_y))
    return c/m

about_the_train = '''\
try to train the model with:
  learning rate: {:f}
  max iteration : {:d}
  cost reduction threshold: {:f}
\
'''
print(about_the_train.format(learning_rate,iteration_count,cost_reduce_threshold))

# start training
cost_last = 0
for i in range(iteration_count):
    grad_w,grad_b = gradient_caculate(w,b)
    w = w - learning_rate*grad_w
    b = b - learning_rate*grad_b
    cost_current = cost_function(w,b,x_train,y_train)
    if i >= iteration_count/2 and cost_last - cost_current<= cost_reduce_threshold:
        print("iteration: {:5d},cost_current:{:f},cost_last:{:f},cost reduce:{:f}".format( i+1,cost_current,cost_last,cost_last-cost_current))
        break
    if (i+1)%(iteration_count/10) == 0:
        print("iteration: {:5d},cost_current:{:f},cost_last:{:f},cost reduce:{:f}".format( i+1,cost_current,cost_last,cost_last-cost_current))
    cost_last = cost_current

print("after the training, parameter w = {:f} and b = {:f}".format(w,b))

for i in range(m):
    y = function_f(x_train[i],w,b)
    p  = 0
    if y>= 0.5: p  = 1
    print("sample: x[{:d}]:{:d},y[{:d}]:{:d}; the prediction is {:d} with probability:{:4f}".format(i,x_train[i],i,y_train[i],p,y))
