# super simple neural network

Learning how machine learning

This repository contains implementations of simple neural networks,include:

* ssnn_ant.py    : a super simple neural network with only one neuron
* ssnn_ant_np.py : a numpy version of ssnn_ant.py
* ssnn_ant_tf.py : a tensorflow version of ssnn_ant.py

* ssnn_bear.py   : a simple neural network with vector input/30+1 neurons/two layer/binary classification

# About ssnn_ant.py

  * only one neuron in only one layer
  * input x is scalar (one-dimension)
  * using logistic function as the activation function
  * more details : [99行代码构建极简的神经网络](https://www.orczhou.com/index.php/2024/11/implement-a-super-simple-neural-network-in-99-line-codes/)

input, parameters and output:

```
input layer:
    x: scalar

parameters:
    w: scalar
    b: scalar

output:
    y \in [0,1] or  p \in {0,1}

```

structure:

```
   x ->   w*x + b   ->   logistic function  -> output
        -----------      -----------------
             |                    |
             V                    V
         one neuron     activation function
```

# How to run ssnn_ant.py

```
python ssnn_ant.py
```

output like :

```
try to train the model with:
  learning rate: 0.010000
  max iteration : 50000
  cost reduction threshold: 0.000001

iteration:  5000,cost_current:0.684258,cost_last:0.684320,cost reduce:0.000062
iteration: 10000,cost_current:0.483588,cost_last:0.483615,cost reduce:0.000027
iteration: 15000,cost_current:0.380745,cost_last:0.380761,cost reduce:0.000016
iteration: 20000,cost_current:0.315210,cost_last:0.315221,cost reduce:0.000011
iteration: 25000,cost_current:0.269129,cost_last:0.269137,cost reduce:0.000008
iteration: 30000,cost_current:0.234776,cost_last:0.234782,cost reduce:0.000006
iteration: 35000,cost_current:0.208126,cost_last:0.208130,cost reduce:0.000005
iteration: 40000,cost_current:0.186834,cost_last:0.186838,cost reduce:0.000004
iteration: 45000,cost_current:0.169431,cost_last:0.169434,cost reduce:0.000003
iteration: 50000,cost_current:0.154942,cost_last:0.154945,cost reduce:0.000003
after the training, parameter w = 5.056985 and b = -22.644516
sample: x[0]:0,y[0]:0; the prediction is 0 with probability:0.000000
sample: x[1]:1,y[1]:0; the prediction is 0 with probability:0.000000
sample: x[2]:2,y[2]:0; the prediction is 0 with probability:0.000004
sample: x[3]:3,y[3]:0; the prediction is 0 with probability:0.000568
sample: x[4]:4,y[4]:0; the prediction is 0 with probability:0.081917
sample: x[5]:5,y[5]:1; the prediction is 1 with probability:0.933417
```

# About ssnn_bear.py

  * input x is matrix 784x1 (where 784 = 28*28 which is MNIST image data)
  * 30 neurons for the only hidden layer, as n^{[1]} = 30
  * output layer: one neuron for classification(logistic)
  * using relu for activation function in hidden layer

input layer:
```
    x: (shape 784x1)
        X: m samples where m = 12665 , X: 784 x 12665
        as : a^{[0]}: 784 x 12665  n^{[0]} = 784
```

hidden layer:
```
    n^{[1]}:  30
    W^{[1]}: (30,784)   as (n^{[1]},n^{[0]})
    Z^{[1]}: (30,12665) as (n^{[1]},m)
    A^{[1]}: (30,12665) as (n^{[1]},m)
```

output layer:
```
    n^{[2]}: 1
    W^{[2]}: (1,30)     as (n^{[2]},n^{[1]})
    Z^{[2]}: (1,12665)  as (n^{[2]},m)
    A^{[2]}: (1,12665)  as (n^{[2]},m)
```

output:
```
    y \in [0,1] or  p \in {0,1}
        Y: (1 x m) ndarray
```

structure for only one/m sample(s):
```
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
```

grad and notaion:
```
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
```

Challenges
    * Understanding the MNIST dataset and labels
    * Understanding gradient caculate and the gradient descent
    * Understanding logistic regression loss function and the caculation
    * Knowing feature normalization

about it:
    it's a simple project for human learning how machine learning
    version ant : scalar input/one neuron/one layer/binary classification
    version bear: vector input/30+1 neurons /two layer/binary classification
    by orczhou.com
