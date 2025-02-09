"""
super simple neural networks (Dolphin version using Adam)
  * [dophin] Using Adam(momentum/RMSp) to accelerate the learning
  * [cat]    Using mini-batch / stochastic gradient descent
  * [bear]   input x is matrix 784x1 (where 784 = 28*28 which is MNIST image data)
  *          30 neurons for the only hidden layer, as n^{[1]} = 30
  * [ant]    output layer: one neuron for classification(logistic)
  *          using relu for activation function in hidden layer

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

mini-batch / stochastic gradient descent:
    for epoch/iteration:
        for a batch :
            mini-batch gradient descent

ADAM(Momentum/RMSprop)
    // momentum
    v = 0
    beta1 = 0.9

    //RMSprop
    s = 0
    beta2 = 0.998

    beta_power_t = 1

    for epoch
        for iteration(one mini-batch)
            beta_power_t = beta_power_t*beta1

            v = beta1*v + (1-beta1)*grad
            v_c = v/(1-beta_power_t)

            s = beta2*v + (1-beta2)*(grad**2)
            s_c = s/(1-beta_power_t)

            w = w - alpha*  v_c/sqrt(s_c)

Challenges
    1. Understanding the MNIST dataset and labels
    2. Understanding gradient caculate and the gradient descent
    3. Understanding logistic regression loss function and the caculation
    3. Knowing feature normalization

about it:
    it's a simple project for human learning how machine learning
    version ant   : scalar input/one neuron/one layer/binary classification
    version bear  : vector input/30+1 neurons /two layer/binary classification
    version cat   : use mini-batch/stochastic gradient descent based on bear version
    version dophin: use ADAM(momentum/rmsp) gradient descent based on cat version
    by orczhou.com
"""

from keras.datasets import mnist
import numpy as np
import math

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
iteration_count = 1
learning_rate   = 0.01
batch_size      = round(m/150)
epsilon = 1e-10

beta1 = 0.9       # momentum
beta2 = 0.998     # rmsp

# mini-batch / stochastic gradient descent(batch size = 1)
# for example:
#    m = 7 , batch_size = 3
#    so:
#       batch_iteration = 7/3 = 3
#       i_batch in 0,1,2
#       in every bath_iteration
#           take sample from
#               i_batch*batch_size
#           to
#               min((i_batch+1)*batch_size,m)
batch_iteration = math.ceil(m/batch_size)

# feature scaling / Normalization
mean = np.mean(X,axis = 1,keepdims = True)
std  = np.std( X,axis = 1,keepdims = True)+epsilon
X  = (X-mean)/std

# initial parameters: W1 W2 b1 b2 size
np.random.seed(561)
W1 = np.random.randn(n1,n0)*0.01
W2 = np.random.randn(n2,n1)*0.01
b1 = np.zeros([n1,1])
b2 = np.zeros([n2,1])

# logistic function
# Note: 145: RuntimeWarning: overflow encountered in exp
def logistic_function(x):
    return 1/(1+np.exp(-x))

about_the_train = '''\
try to train the model with:
  learning rate: {:f}
  iteration    : {:d}
  neurons in hidden layer: {:d}
  batch_size :   {:.2f}
  momentum:      {:.1f}
  RMSp:          {:.3f}
\
'''
print(about_the_train.format(learning_rate,iteration_count,n1,batch_size,beta1,beta2))

# i for epoch;   i_batch_cnt++: every i_batch iteration
i_batch_cnt = 0
adam_v_W1,adam_v_W2,adam_v_b1,adam_v_b2 = 0,0,0,0         # momentum    v_0 = 0
adam_s_W1,adam_s_W2,adam_s_b1,adam_s_b2 = 0,0,0,0         # rmsp        s_0 = 0
beta1_power_t = 1     # beta1^t = 1 when t = 0
beta2_power_t = 1     # beta2^t = 1 when t = 0
def update_parameters_adam(grad,para,adam_v_para,adam_s_para,beta1_power_t,beta2_power_t):
    adam_v_para   = beta1*adam_v_para + (1-beta1)*grad
    adam_v_para_c = adam_v_para/(1-beta1_power_t)
    adam_s_para   = beta2*adam_s_para + (1-beta2)*(grad**2)
    adam_s_para_c = adam_s_para/(1-beta2_power_t)
    para = para - learning_rate*adam_v_para_c/(np.sqrt(adam_s_para_c)+epsilon)
    return (para,adam_v_para,adam_s_para)

cost_last = np.inf # very large data,maybe better , what about np.inf
for i in range(iteration_count):
    print("iteration/epoch: {:5d}".format( i+1))
    for i_batch in range(batch_iteration):
        i_batch_cnt += 1
        # sample from i_batch*batch_size
        # forward/backward propagation (read the comment above:"grad and notaion")
        batch_from  = i_batch*batch_size
        batch_to    = min((i_batch+1)*batch_size,m)
        batch_count = batch_to - batch_from
        A0  = X[:,batch_from:batch_to] # column  [batch_from,batch_to)
        Y_L = Y[:,batch_from:batch_to] # Y_label [batch_from,batch_to)

        Z1 = W1@A0 + b1       # W1 (n1,n0)  X: (n0,m)
        A1 = np.maximum(Z1,0) # relu
        Z2 = W2@A1 + b2
        A2 = logistic_function(Z2)

        dZ2 = (A2-Y_L)/batch_count
        dW2 = dZ2@A1.T
        db2 = np.sum(dZ2,axis=1,keepdims = True)

        dZ1 = W2.T@dZ2*(np.where(Z1 > 0, 1, 0)) # np.where(Z1 > 0, 1, 0) is derivative of relu function
        dW1 = dZ1@A0.T
        db1 = np.sum(dZ1,axis=1,keepdims = True)

        # Note: 183: RuntimeWarning: divide by zero encountered in log
        # it seems that it happens a lot
        #     think about that if for some sample a2 = 0/1
        #     "Y_L*(np.log(A2))" or "(np.log(1-A2))" will encounter will like above
        A2_clip = np.clip(A2, epsilon, 1 - epsilon)
        cost_current = np.sum(-(Y_L*(np.log(A2_clip))) - ((1-Y_L)*(np.log(1-A2_clip))))/batch_count
        # if (i+1)%(iteration_count/20) == 0 and (i_batch+1)%(batch_iteration/20) ==0 :
        print("batch iteration: {:5d},cost_current:{:f},cost_last:{:f},cost reduce:{:f}".format( i_batch_cnt,cost_current,cost_last,cost_last-cost_current))

        cost_last = cost_current
        '''
            for epoch
                for iteration(one mini-batch)
                    beta1_power_t = beta1_power_t*beta1
                    beta2_power_t = beta2_power_t*beta2

                    v = beta1*v + (1-beta1)*grad
                    v_c = v/(1-beta1_power_t)

                    s = beta2*v + (1-beta2)*(grad**2)
                    s_c = s/(1-beta2_power_t)

                    w = w - alpha*  v_c/sqrt(s_c)
        '''
        beta1_power_t = beta1_power_t*beta1
        beta2_power_t = beta2_power_t*beta2

        (W1,adam_v_W1,adam_s_W1) = update_parameters_adam(dW1,W1,adam_v_W1,adam_s_W1,beta1_power_t,beta2_power_t)
        (W2,adam_v_W2,adam_s_W2) = update_parameters_adam(dW2,W2,adam_v_W2,adam_s_W2,beta1_power_t,beta2_power_t)
        (b1,adam_v_b1,adam_s_b1) = update_parameters_adam(db1,b1,adam_v_b1,adam_s_b1,beta1_power_t,beta2_power_t)
        (b2,adam_v_b2,adam_s_b2) = update_parameters_adam(db2,b2,adam_v_b2,adam_s_b2,beta1_power_t,beta2_power_t)

        # W1 = W1 - learning_rate*dW1
        # W2 = W2 - learning_rate*dW2
        # b1 = b1 - learning_rate*db1
        # b2 = b2 - learning_rate*db2


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
