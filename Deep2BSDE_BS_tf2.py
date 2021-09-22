#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
 BS modelに変更したプログラム
 time discretizationの刻み（N）を増やすと処理時間が増える。(MLPの数/パラメータも増える)
 収束速度は不安定-> strike/spotのスケールを100で割って調整した方が収束が速くBatch Normalizationがワークする。
"""

import time, datetime
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
import tensorflow_probability as tfp

start_time = time.time()

name = 'BS_TF2_ADJ'
d = 1
batch_size = 64
T = 1.0
N = 125
h = T/N
sqrth = np.sqrt(h)
n_maxstep = 1000
n_displaystep = 50
SCALE = 100
Xinit = np.array([100]*d)/SCALE
mu = 0
sigma = 0.1
K = 100.0/SCALE
r = 0.05

def f_tf(t, X, Y, Z, Gamma):
    return -0.5*(sigma**2)*tf.expand_dims(tf.linalg.trace(
        tf.square(tf.expand_dims(X,-1)) * Gamma),-1) + \
                r * (Y - tf.math.reduce_sum(X*Z, 1, keepdims = True)) 

def _opt(X_):
    a = tf.zeros([batch_size, 1], dtype=tf.float32)
    X_ = tf.where(X_ < a, a, X_)
    return X_

class Dense(tf.Module):
    def __init__(self, input_dim, output_size, activation, is_last, name=None):
        super(Dense, self).__init__(name=name)
        with self.name_scope:
            self.w = tf.Variable(
                tf.random.normal([input_dim, output_size],mean=0,stddev=5/np.sqrt(input_dim+output_size)), name='w')
            self.b = tf.Variable(tf.zeros([output_size]), name='b')
            self.BN = tf.keras.layers.BatchNormalization()
            self.activation = activation
            self.is_last = is_last
    @tf.Module.with_name_scope
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.is_last == False:
            y = self.BN(y)
        if self.activation == "ReLU":
            y = tf.nn.relu(y)
        return y

class MLP(tf.Module):
    def __init__(self, input_size, sizes, activations, name=None):
        super(MLP, self).__init__(name=name)
        self.layers = []
        with self.name_scope:
            for _k, (size, act_fun) in enumerate(zip(sizes,activations)):
                self.layers.append(Dense(input_dim=input_size, 
                                         output_size=size,
                                         activation=act_fun,
                                         is_last=(_k == (len(sizes)-1))
                                         ))
                input_size = size
    @tf.Module.with_name_scope
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class forward(tf.Module):
    def __init__(self,name=None):
        super(forward, self).__init__(name=name)
        self.mlps = []
        with self.name_scope:
            for t in range(N-1):
                mlps = []
                mlps.append(MLP(input_size=d, sizes=[d,d,d],activations=["ReLU","ReLU","Linear"]))
                mlps.append(MLP(input_size=d, sizes=[d,d,d**2],activations=["ReLU","ReLU","Linear"]))
                self.mlps.append(mlps)
    @tf.Module.with_name_scope
    def __call__(self, X, Y, Z, A, Gamma):
        for t,mlps in zip(range(N-1),self.mlps):
            
            dW = tf.random.normal(shape=[batch_size, d], stddev = 1, dtype=tf.float32)
            
            # Y update inside the loop
            dX = r * h * X + \
                    sqrth * sigma * X * dW

            Y = Y + f_tf(t * h, X, Y, Z, Gamma)*h \
                    + tf.reshape(tf.linalg.trace(Gamma), [batch_size, 1])*tf.square(sigma)*(X**2)* h*0.5 \
                    + tf.reduce_sum(Z*dX, 1, keepdims = True)
                    
            Z = Z + A * h + tf.squeeze(tf.matmul(Gamma, tf.expand_dims(dX, -1), transpose_b=False))
            
            X = X + dX
            
            A = mlps[0](X)/d
            Gamma = mlps[1](X)/d**2
            Gamma = tf.reshape(Gamma, [batch_size, d, d])
            
        dW = tf.random.normal(shape=[batch_size, d], stddev = 1, dtype=tf.float32)
            
        # Y update inside the loop
        dX = r * h * X + sqrth * sigma * X * dW

        Y = Y + f_tf((N-1) * h, X, Y, Z, Gamma)*h \
                + tf.reshape(tf.linalg.trace(Gamma), [batch_size, 1])*tf.square(sigma)*(X**2)* h*0.5 \
                + tf.reduce_sum(Z*dX, 1, keepdims = True)

        X = X + dX
            
        loss = tf.reduce_mean(tf.square(Y-_opt(X-K)*SCALE))

        return loss


class Deep2BSDE(tf.Module):
    def __init__(self, name=None):
        super(Deep2BSDE, self).__init__(name=name)
        with self.name_scope:
            self.X = tf.Variable(np.ones([batch_size, d]) * Xinit,
                                dtype=tf.float32,
                                trainable=False)
            self.Y0 = tf.Variable(tf.random.uniform([1],
                                                    minval=0, maxval=50, dtype=tf.float32),
                                name='Y0')
            self.Z0 = tf.Variable(tf.random.uniform([1, d],
                                                    minval=-.1, maxval=.1,
                                                    dtype=tf.float32),
                                name='Z0')
            self.Gamma0 = tf.Variable(tf.random.uniform([d,d],
                                                minval=-1, maxval=1,
                                                dtype=tf.float32),
                                    name='Gamma0')
            self.A0 = tf.Variable(tf.random.uniform([1, d], 
                                                    minval=-.1, maxval=.1,
                                                    dtype=tf.float32), 
                                name='A0')
            self.fwd = forward()
    @tf.Module.with_name_scope
    def __call__(self):
        allones = tf.ones(shape=[batch_size, 1], 
                            dtype=tf.float32,
                            name='MatrixOfOnes')
        Y = allones * self.Y0
        Z = tf.matmul(allones, self.Z0)
        A = tf.matmul(allones, self.A0)
        
        Gamma = tf.multiply(tf.ones(shape=[batch_size, d, d],
                                        dtype=tf.float32), 
                                self.Gamma0)
        
        loss = self.fwd(self.X,Y,Z,A,Gamma)
        return loss
    

model = Deep2BSDE()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.9)

@tf.function
def train_step():
  with tf.GradientTape() as tape:
    loss  = model()
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  return loss

y0_values = []
steps = []
losses = []
running_time = []

for epoch in range(n_maxstep):
    loss = train_step()
    Y = model.trainable_variables[2].numpy()
    template = 'Epoch {}, Loss: {:.0f}, y0: {:.3f}'
    if ((epoch+1) % n_displaystep) ==0:
        steps.append(epoch+1)
        losses.append(loss)
        y0_values.append(Y)
        running_time.append(time.time()-start_time)
        print(template.format(epoch+1,
                              loss,Y[0]
                              )
              )

output = np.zeros((len(y0_values),4))
output[:,0] = steps
output[:,1] = losses
output[:,2] = y0_values
output[:,3] = running_time
np.savetxt("output/"+str(name) + "_d" + str(d) + "_" + \
    datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".csv",
    output,
    delimiter = ",",
    header = "step, loss function, Y0, running time"
    )