#!/usr/bin/env python3
import numpy as np
from scipy import interpolate

import tensorflow as tf

def forward_solution(data_input, n_samples, state_0, n_steps, num_outputs, dt, rhs):
    x = state_0

    y_history = tf.TensorArray(tf.float64, size = n_steps)
    y_history = y_history.write(0, x[:, :num_outputs])
    for i in tf.range(n_steps - 1):
        x = x + dt * rhs(x, data_input[:, i, :]) 
        y_history = y_history.write(i + 1, x[:, :num_outputs])
    return tf.transpose(y_history.stack(), perm = (1, 0, 2))

def forward_solution_ic(data_input, n_samples, y0, z0, n_steps, num_outputs, dt, rhs):
    x = tf.concat((y0, z0), axis = 1)

    y_history = tf.TensorArray(tf.float64, size = n_steps)
    y_history = y_history.write(0, x[:, :num_outputs])
    for i in tf.range(n_steps - 1):
        x = x + dt * rhs(x, data_input[:, i, :]) 
        y_history = y_history.write(i + 1, x[:, :num_outputs])
    return tf.transpose(y_history.stack(), perm = (1, 0, 2))

def build_NN(layers, input_shape, num_outputs):
    lay_list = []
    lay_list.append(tf.keras.layers.Dense(layers[0], input_shape = input_shape, activation = tf.nn.tanh))
    for i in range(1, len(layers)):
        lay_list.append(tf.keras.layers.Dense(layers[i], activation = tf.nn.tanh))
    lay_list.append(tf.keras.layers.Dense(num_outputs))
    return tf.keras.Sequential(lay_list)

def adim(v, v_min, v_max):
    return (2.0 * v - v_min - v_max) / (v_max - v_min)

def redim(V, v_min, v_max):
    return 0.5 * (v_min + v_max + (v_max - v_min) * V)

def reshape_min_max(n, v_min, v_max, axis = None):
    if axis is not None:
        shape_min = [1] * n
        shape_max = [1] * n
        shape_min[axis] = len(v_min)
        shape_max[axis] = len(v_max)
        v_min = np.reshape(v_min, shape_min)
        v_max = np.reshape(v_max, shape_max)
    return v_min, v_max

def normalize_forw(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return (2.0 * v - v_min - v_max) / (v_max - v_min)

def normalize_back(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return 0.5 * (v_min + v_max + (v_max - v_min) * v)

def fix_shape(v, n):
    v = np.ones(n) * np.array(v)
    return np.array([v])

def flatten_list_array(_list_array):
    flat_list = []
    for element in _list_array:
        if isinstance(element, list) or isinstance(element, np.ndarray):
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

class GP_generator:
    def __init__(self, x, mean, std, correlation_length):
        self.x = x
        self.mean = mean
        self.std = std
        self.correlation_length = correlation_length
        self.initialize()
    
    def initialize(self):
        self.mu = [self.mean for x1 in self.x]
        self.Sigma = [[self.kernel(x1, x2) for x2 in self.x] for x1 in self.x]

    def kernel(self, x1, x2):
        return self.std * self.std * np.exp(-1 * ((x1 - x2) * (x1 - x2)) / (2 * self.correlation_length * self.correlation_length))

    def get_sample(self, interp = False):
        vals = np.random.multivariate_normal(self.mu, self.Sigma)
        if interp:
            return interpolate.interp1d(self.x, vals, fill_value = 'extrapolate')
        else:
            return vals

    def get_sigma(self):
        return np.reshape(np.array(self.Sigma), (np.size(self.x), np.size(self.x)))