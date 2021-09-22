


"""
Beck et al. 2019 Appendix A.1 Original Program
#! /usr/bin/python3
 f/gammaの修正 : 39 / 113 / 128行目の修正 
 Plain deep2BSDE solver with hard-coded Allen-Cahn equation
"""

import tensorflow as tf
import numpy as np
import time, datetime

tf.compat.v1.reset_default_graph()
start_time = time.time()

name = 'AllenCahn'

# setting of the problem
d = 20
T = 0.3
Xi = np.zeros([1, d])

# setup of algorithm and implementation
N = 20
h = T/N
sqrth = np.sqrt(h)
n_maxstep = 10000
batch_size = 1
gamma = 0.001

# neural net architectures
n_neuronForGamma = [d, d, d, d**2]
n_neuronForA = [d, d, d, d]

# (adapted) rhs of the pde
def f0(t, X, Y, Z, Gamma):
    return -Y+tf.pow(Y, 3)-0.5*Z 

# terminal condition
def g(X):
    return 1/(1 + 0.2*tf.compat.v1.reduce_sum(tf.square(X), 1, keep_dims=True))*0.5


# helper functions for constructing the neural net(s)
def _one_time_net(x, name, isgamma=False):
    with tf.compat.v1.variable_scope(name):
        layer1 = _one_layer(x, (1-isgamma)*n_neuronForA[1] +
                            isgamma*n_neuronForGamma[1], name='layer1')
        layer2 = _one_layer(layer1, (1-isgamma)*n_neuronForA[2] +
                            isgamma*n_neuronForGamma[2], name='layer2')
        z = _one_layer(layer2, (1-isgamma)*n_neuronForA[3] +
                       isgamma*n_neuronForGamma[3], activation_fn=None,
                       name='final')
        return z

def _one_layer(input_, output_size, activation_fn=tf.nn.relu,
               stddev=5.0, name='linear'):
    with tf.compat.v1.variable_scope(name):
        shape = input_.get_shape().as_list()
        w = tf.compat.v1.get_variable('Matrix', [shape[1], output_size],
                            tf.float64,
                            tf.compat.v1.random_normal_initializer(
                            stddev=stddev/np.sqrt(shape[1]+output_size)))
        b = tf.compat.v1.get_variable('Bias', [1, output_size], tf.float64,
                            tf.constant_initializer(0.0))
        hidden = tf.matmul(input_, w) + b
        if activation_fn:
            return activation_fn(hidden)
        else:
            return hidden

with tf.compat.v1.Session() as sess:
    # background dynamics
    dW = tf.compat.v1.random_normal(shape=[batch_size, d], stddev=sqrth,
                          dtype=tf.float64)

    # initial values of the stochastic processes
    X = tf.compat.v1.Variable(np.ones([batch_size, d]) * Xi,
                    dtype=tf.float64,
                    name='X',
                    trainable=False)
    Y0 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([1],
                     minval=-1, maxval=1,
                     dtype=tf.float64),
                     name='Y0')
    Z0 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([1, d],
                     minval=-.1, maxval=.1,
                     dtype=tf.float64),
                     name='Z0')
    Gamma0 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([d, d],
                         minval=-.1, maxval=.1,
                         dtype=tf.float64),
                         name='Gamma0')
    A0 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([1, d],
                     minval=-.1, maxval=.1,
                     dtype=tf.float64),
                     name='A0')
    allones = tf.ones(shape=[batch_size, 1],
                      dtype=tf.float64,
                      name='MatrixOfOnes')
    Y = allones * Y0
    Z = tf.matmul(allones, Z0)
    Gamma = tf.multiply(tf.ones([batch_size, d, d],
                        dtype=tf.float64), Gamma0)
    A = tf.matmul(allones, A0)

    # forward discretization
    with tf.compat.v1.variable_scope('forward'):
        for i in range(N-1):
            Y = Y + f0(i*h, X, Y, Z, Gamma)*h \
                + tf.reshape(tf.compat.v1.trace(Gamma)*2, [batch_size, 1])*h*0.5 \
                + tf.compat.v1.reduce_sum(dW*Z, 1, keep_dims=True)
            Z = Z + A * h \
                + tf.squeeze(tf.matmul(Gamma,
                                       tf.expand_dims(dW, -1)))
            Gamma = tf.reshape(_one_time_net(X, name=str(i)+'Gamma',
                                             isgamma=True)/d**2,
                               [batch_size, d, d])
            if i != N-1:
                A = _one_time_net(X, name=str(i)+'A')/d
            X = X + dW
            dW = tf.compat.v1.random_normal(shape=[batch_size, d],
                                  stddev=sqrth, dtype=tf.float64)

        Y = Y + f0((N-1)*h, X, Y, Z, Gamma)*h \
            + tf.reshape(tf.compat.v1.trace(Gamma)*2, [batch_size, 1])*h*0.5 \
            + tf.compat.v1.reduce_sum(dW*Z, 1, keep_dims=True)
        X = X + dW
        loss_function = tf.reduce_mean(tf.square(Y-g(X)))

    # specifying the optimizer
    global_step = tf.compat.v1.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False, dtype=tf.int32)

    learning_rate = tf.compat.v1.train.exponential_decay(gamma, global_step,
                                               decay_steps=10000,
                                               decay_rate=0.0, staircase=True)
    trainable_variables = tf.compat.v1.trainable_variables()
    grads = tf.gradients(loss_function, trainable_variables)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                                                  learning_rate=learning_rate)
    apply_op = optimizer.apply_gradients(
                                    zip(grads, trainable_variables),
                                    global_step=global_step, name='train_step')

    with tf.control_dependencies([apply_op]):
        train_op_2 = tf.identity(loss_function, name='train_op2')

    # to save history
    learning_rates = []
    y0_values = []
    losses = []
    running_time = []
    steps = []
    sess.run(tf.compat.v1.global_variables_initializer())

    try:
        # the actual training loop
        for _ in range(n_maxstep + 1):
            y0_value, step = sess.run([Y0, global_step])
            currentLoss, currentLearningRate = sess.run(
                [train_op_2, learning_rate])

            learning_rates.append(currentLearningRate)
            losses.append(currentLoss)
            y0_values.append(y0_value)
            running_time.append(time.time()-start_time)
            steps.append(step)

            if step % 100 == 0:
                print("step: ", step,
                      " loss: ", currentLoss,
                      " Y0: ", y0_value,
                      " learning rate: ", currentLearningRate)

        end_time = time.time()
        print("running time: ", end_time-start_time)

    except KeyboardInterrupt:
        print("\nmanually disengaged")

# writing results to a csv file
output = np.zeros((len(y0_values), 5))
output[:, 0] = steps
output[:, 1] = losses
output[:, 2] = y0_values
output[:, 3] = learning_rates
output[:, 4] = running_time

np.savetxt("output/"+str(name) + "_d" + str(d) + "_" +
           datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+".csv",
           output,
           delimiter=",",
           header="step, loss function, Y0, learning rate, running time",
           )
