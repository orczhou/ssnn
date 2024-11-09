"""
super simple neural networks 
  * only one neuron in only the one hidden layer
  * input x is scalar (one-dimension)
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
# x   : scalar
# w   : scalar
# b   : scalar
def function_f(x,w,b):  
    return 1/(1+math.exp(-(x*w+b)))

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
# x_p: x_train
# y_p: y_train
# w_p: current w
# b_p: current b
def gradient_caculate(x_p,y_p,w_p,b_p):
    gradient_w,gradient_b = (0.,0.)
    for i in range(m):
        gradient_w += x_p[i]*(function_f(x_p[i],w_p,b_p)-y_p[i])
        gradient_b += function_f(x_p[i],w_p,b_p)-y_p[i]
    return gradient_w,gradient_b

def cost_function(w_p,b_p,x_p,y_p):
    c = 0
    for i in range(m):
        y = function_f(x_p[i],w_p,b_p)
        c += -y_p[i]*math.log(y) - (1-y_p[i])*math.log(1-y)
    return c

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
    grad_w,grad_b = gradient_caculate(x_train,y_train,w,b)
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
