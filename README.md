# super simple neural network

Learning how machine learning

This repository contains implementations of simple neural networks,include:

* ssnn_ant.py    : a super simple neural network with only one neuron
* ssnn_ant_np.py : a numpy version of ssnn_ant.py
* ssnn_ant_tf.py : a tensorflow version of ssnn_ant.py

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
