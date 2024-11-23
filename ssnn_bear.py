"""
super simple neural networks (bear version)
  * input x is matrix 784x1 (where 784 = 28*28 which is MNIST image data)
  * 30 neurons for the only hidden layer, as n^{[1]} = 30
  * output layer: one neuron for classification(logistic)
  * using relu for activation function in hidden layer

input layer:
    x: (shape 784x1)
        X: m samples where m = 12665 , X: 784 x 12665
        as : a^{[0]}: 784 x 12665  n^{[0]} = 784
hidden layer:
    n^{[1]}:  30
    W^{[1]}: (30,784)   as (n^{[1]},n^{[0]})
    Z^{[1]}: (30,12665) as (n^{[1]},m)
    A^{[1]}: (30,12665) as (n^{[1]},m)
output layer:
    n^{[2]}: 1
    W^{[2]}: (1,30)     as (n^{[2]},n^{[1]})
    Z^{[2]}: (1,12665)  as (n^{[2]},m)
    A^{[2]}: (1,12665)  as (n^{[2]},m)

output:
    y \in [0,1] or  p \in {0,1}
        Y: (1 x m) ndarray
structure for only one sample:
      x_1   ->   W*X + B   ->  relu  ->
      x_2   ->   W*X + B   ->  relu  ->  \
      ...   ->     ...     ->     .. ->  -> w*x+b -> logistic
      x_784 ->   W*X + B   ->  relu  ->  /
     ------     --------------------       ------------------
       |                |                          |
       V                V                          V
     input         30 neurons                 one neuron
    feature      relu activation             output layer

  By numpy with m samples:
    np.logistic(W2@g(W1@X+b1)+b2) as \hat{Y}: (1 x m) ndarray

    dimension analysis:
        W2        : (n2,n1)
        g(W1@X+b1): (n1,m)
            W1 : (n1,n0)
            X  : (n0,m)
            b1 : (n1,1)  with broadcasting to (n1,m)
        b2: (n2,1) with broadcasting to (n2,m)

grad and notaion:
    forward propagation : A1 A2 Z1 Z2
    backward propagation: dW1 dW2 db1 db2

    more details:
        Z1 = W1@X  + b1
        Z2 = W2@A1 + b2
        A1 = g(Z1)      -- g     for relu
        A2 = \sigma(Z2) -- sigma for logistic

        dW2 = ((1/m)*(A2-Y))@A1.T
            dW2 = dZ2@A1.T  where dZ2 = (1/m)*(A2-Y)
            A2.shape:(1,m) Y.shape:(1,m) A1.T.shape:(n1,m)
            so: dW2.shape: (1,n1)

        dW1 = (W2.T@((1/m)*(A2-Y))*g_prime(Z1))@A0.T
            dW1 = dZ1@A1.T
                where
                    dZ1 = W2.T@dZ2 * g_prime(Z1)
                    g_prime is derivative of relu
                dW2.shape: (n1,n0)
        note: @ for matrix multiply;   * for dot product/element-wise

Challenges
    1. Understanding the MNIST dataset and labels
    2. Understanding gradient caculate and the gradient descent
    3. Understanding logistic regression loss function and the caculation
    3. Knowing feature normalization

about it:
    it's a simple project for human learning how machine learning
    version ant : scalar input/one neuron/one layer/binary classification
    version bear: vector input/30+1 neurons /two layer/binary classification
    by orczhou.com
"""

from keras.datasets import mnist
import numpy as np

# return only data lable 0 or 1 from MNIST for the binary classification
def filter_mnist_data(data_X, data_y):
    data_filter = np.where((data_y == 0) | (data_y == 1))
    filtered_data_X, filtered_data_y = data_X[data_filter], data_y[data_filter]
    r_data_X = filtered_data_X.reshape(filtered_data_X.shape[0],filtered_data_X.shape[1]*filtered_data_X.shape[2])
    return (r_data_X, filtered_data_y)

(train_all_X, train_all_y), (test_all_X, test_all_y) = mnist.load_data()
(train_X,train_y) = filter_mnist_data(train_all_X, train_all_y)
(test_X ,test_y ) = filter_mnist_data(test_all_X, test_all_y)

X  = train_X.T
Y  = train_y.reshape(1,train_y.shape[0])

m = X.shape[1]    # number of samples

# hyper-parameter; read the comments above for structure of the NN
n0 = X.shape[0]   # number of input features
n1 = 10           # nerons of the hidden layer
n2 = 1            # nerons of the output layer
iteration_count = 150
learning_rate   = 0.1
epsilon = 1e-10

# feature scaling / Normalization
mean = np.mean(X,axis = 1,keepdims = True)
std  = np.std(X,axis = 1,keepdims = True)+epsilon
X  = (X-mean)/std

# initial parameters: W1 W2 b1 b2 size
np.random.seed(561)
W1 = np.random.randn(n1,n0)*0.01
W2 = np.random.randn(n2,n1)*0.01
b1 = np.zeros([n1,1])
b2 = np.zeros([n2,1])

# logistic function
def logistic_function(x):
    return 1/(1+np.exp(-x))

about_the_train = '''\
try to train the model with:
  learning rate: {:f}
  iteration    : {:d}
  neurons in hidden layer: {:d}
\
'''
print(about_the_train.format(learning_rate,iteration_count,n1))

# forward/backward propagation (read the comment above:"grad and notaion")
cost_last = np.inf # very large data,maybe better , what about np.inf
for i in range(iteration_count):
    # forward propagation
    A0 = X

    Z1 = W1@X + b1  # W1 (n1,n0)  X: (n0,m)
    A1 = np.maximum(Z1,0) # relu
    Z2 = W2@A1 + b2
    A2 = logistic_function(Z2)

    dZ2 = (A2-Y)/m
    dW2 = dZ2@A1.T
    db2 = np.sum(dZ2,axis=1,keepdims = True)

    dZ1 = W2.T@dZ2*(np.where(Z1 > 0, 1, 0)) # np.where(Z1 > 0, 1, 0) is derivative of relu function
    dW1 = dZ1@A0.T
    db1 = np.sum(dZ1,axis=1,keepdims = True)

    cost_current = np.sum(-(Y*(np.log(A2))) - ((1-Y)*(np.log(1-A2))))/m
    if (i+1)%(iteration_count/150) == 0:
        print("iteration: {:5d},cost_current:{:f},cost_last:{:f},cost reduce:{:f}".format( i+1,cost_current,cost_last,cost_last-cost_current))

    cost_last = cost_current
    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2

print("Label:")
print(np.round( Y[0][:20]+0.,0))
print("Predict:")
print(np.round(A2[0][:20],0))

# Normalization for test dataset
X  = (test_X.T - mean)/std
Y  = test_y.reshape(1,test_y.shape[0])

Y_predict = (logistic_function(W2@np.maximum((W1@X+b1),0)+b2) > 0.5).astype(int)

for index in (np.where(Y != Y_predict)[1]):
    print(f"failed to recognize: {index}")
    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(linewidth=np.inf)
    # print(test_X[index].reshape(28,28))

print("total test set:" + str(Y.shape[1]) + ",and err rate:"+str((np.sum(np.square(Y-Y_predict)))/Y.shape[1]))
