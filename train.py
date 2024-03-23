#!/usr/bin/env python3
import os
import pickle
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
import scipy.stats

import tensorflow as tf

import LNODEs
from utilities import *

# Set double precision in TensorFlow.
tf.keras.backend.set_floatx('float64')

# Set automatic number of threads in TensorFlow.
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

# Define loss function(s).
def get_rhs_weights_MS():
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in rhs_base.layers]) / len(rhs_base.layers)

def fit_train():
    out_train = forward_solution_ic(data_ip_train, n_samples_train, y0_train, z0_train, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(out_train - data_op_train), axis = 1), axis = 0) / norm_train)
    return loss

def fit_train_max():
    out_train = forward_solution_ic(data_ip_train, n_samples_train, y0_train, z0_train, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(tf.reduce_max(out_train, axis = 1) - tf.reduce_max(data_op_train, axis = 1)), axis = 0) / norm_train_max)
    return loss

def fit_train_min():
    out_train = forward_solution_ic(data_ip_train, n_samples_train, y0_train, z0_train, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(tf.reduce_min(out_train, axis = 1) - tf.reduce_min(data_op_train, axis = 1)), axis = 0) / norm_train_min)
    return loss

def fit_train_diff():
    out_train = forward_solution_ic(data_ip_train, n_samples_train, y0_train, z0_train, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(tf.experimental.numpy.diff(out_train, axis = 1) - tf.experimental.numpy.diff(data_op_train, axis = 1)), axis = 1), axis = 0) / norm_train_diff)
    return loss

def fit_test():
    out_test = forward_solution(data_ip_test, n_samples_test, state_0_test, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(out_test - data_op_test), axis = 1), axis = 0) / norm_test)
    return loss

def fit_test_max():
    out_test = forward_solution(data_ip_test, n_samples_test, state_0_test, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(tf.reduce_max(out_test, axis = 1) - tf.reduce_max(data_op_test, axis = 1)), axis = 0) / norm_test_max)
    return loss

def fit_test_min():
    out_test = forward_solution(data_ip_test, n_samples_test, state_0_test, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(tf.reduce_min(out_test, axis = 1) - tf.reduce_min(data_op_test, axis = 1)), axis = 0) / norm_test_min)
    return loss

def fit_test_diff():
    out_test = forward_solution(data_ip_test, n_samples_test, state_0_test, n_steps, num_out, dt, rhs)
    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(tf.experimental.numpy.diff(out_test, axis = 1) - tf.experimental.numpy.diff(data_op_test, axis = 1)), axis = 1), axis = 0) / norm_test_diff)
    return loss

# LNODE parameters.
neurons = 13
layers = 3
states = 8
weight_w_rhs = 0.023
weight_min = 0.1
weight_max = 0.1
weight_diff = 0.1
dt = 1e-3 # [s].
dt_base = 0.0285 # [s].
num_iters_ADAM = 1000
num_iters_BFGS = 10000

# Input/Output.
training_folder = './data/'
output_folder_base = './ANNs/'

# Set seed for reproducibility.
np.random.seed(1)
tf.random.set_seed(1)

# Import dataset.
input_file = open(training_folder + 'simulations.pkl', "rb")
dataset = pickle.load(input_file)

# Inputs.
num_inp = dataset['inp'].shape[2]
inp_min = np.min(dataset['inp'], axis = (0, 1)).tolist()
inp_max = np.max(dataset['inp'], axis = (0, 1)).tolist()    
inp_min = fix_shape(inp_min, num_inp)
inp_max = fix_shape(inp_max, num_inp)

# Outputs.
num_out = dataset['out'].shape[2]
out_min = np.min(dataset['out'], axis = (0, 1)).tolist()
out_max = np.max(dataset['out'], axis = (0, 1)).tolist()
out_min = fix_shape(out_min, num_out)
out_max = fix_shape(out_max, num_out)

# Samples, times, variables.
num_samples = dataset['inp'].shape[0]
num_times = dataset['inp'].shape[1]
dataset['times'] = dataset['times'] / 1000. # From ms to s.
THB = 0.854 # [s].
last_HB = int(3 * 1000 * THB)
times = np.arange(dataset['times'][last_HB], dataset['times'][-1] + dt * 1e-10, step = dt)
n_steps = len(times)

num_states = int(states)

if num_states < num_out:
    raise Exception('Not allowed num_states < num_out')

state_min = fix_shape(-1, num_states)
state_max = fix_shape(+1, num_states)
state_min[:,:num_out] = out_min
state_max[:,:num_out] = out_max

test_indices = [262, 365, 230, 10, 33]
train_indices = [x for x in range(0, 405) if x not in test_indices]

n_samples_train = len(train_indices)
n_samples_test = len(test_indices)

output_folder = output_folder_base + 'rhs_' \
              + str(n_samples_train) + 'train_' + str(n_samples_test) + 'test_' \
              + str(int(neurons)) + 'neur_' + str(int(layers)) + 'lay_' \
              + str(int(states)) + 'states_' \
              + str(round(weight_w_rhs, 4)) + 'L2reg_' \
              + str(round(dt_base, 4)) + 'dtbase' \
              + '/'
os.makedirs(output_folder, exist_ok = True)

data_ip_train = interpolate.interp1d(dataset['times'], dataset['inp'][train_indices, :, :], axis = 1)(times)
data_op_train = interpolate.interp1d(dataset['times'], dataset['out'][train_indices, :, :], axis = 1)(times)
data_ip_test = interpolate.interp1d(dataset['times'], dataset['inp'][test_indices, :, :], axis = 1)(times)
data_op_test = interpolate.interp1d(dataset['times'], dataset['out'][test_indices, :, :], axis = 1)(times)

# Multiple initial conditions.
y0_train = data_op_train[:, 0, :num_out]
y0_train = tf.constant(y0_train)
z0_train = tf.Variable(np.zeros((n_samples_train, num_states - num_out)))

y0_test = data_op_test[:, 0, :num_out]
state_0_test = np.zeros((n_samples_test, num_states))
state_0_test[:, :num_out] = y0_test
state_0_test = tf.constant(state_0_test)

data_ip_train = tf.convert_to_tensor(data_ip_train)
data_op_train = tf.convert_to_tensor(data_op_train)
data_ip_test = tf.convert_to_tensor(data_ip_test)
data_op_test = tf.convert_to_tensor(data_op_test)
    
# Define right hand side.
rhs_base = build_NN(int(layers) * [int(neurons)], (num_states + num_inp, ), num_states)
rhs = lambda state, inputs: redim(rhs_base(tf.concat([adim(state, state_min, state_max), adim(inputs, inp_min, inp_max)], axis = 1)), state_min, state_max) / dt_base

# Compute normalizations on training set.
norm_train = tf.reduce_mean(tf.reduce_mean(tf.square(data_op_train), axis = 1), axis = 0)
norm_train_max = tf.reduce_mean(tf.square(tf.reduce_max(data_op_train, axis = 1)), axis = 0)
norm_train_min = tf.reduce_mean(tf.square(tf.reduce_min(data_op_train, axis = 1)), axis = 0)
norm_train_diff = tf.reduce_mean(tf.reduce_mean(tf.square(tf.experimental.numpy.diff(data_op_train, axis = 1)), axis = 1), axis = 0)

# Compute normalization(s) on testing set.
norm_test = tf.reduce_mean(tf.reduce_mean(tf.square(data_op_test), axis = 1), axis = 0)
norm_test_max = tf.reduce_mean(tf.square(tf.reduce_max(data_op_test, axis = 1)), axis = 0)
norm_test_min = tf.reduce_mean(tf.square(tf.reduce_min(data_op_test, axis = 1)), axis = 0)
norm_test_diff = tf.reduce_mean(tf.reduce_mean(tf.square(tf.experimental.numpy.diff(data_op_test, axis = 1)), axis = 1), axis = 0)

# Combine loss function(s).
def loss_fit_train(): return fit_train() + weight_max * fit_train_max() + weight_min * fit_train_min() + weight_diff * fit_train_diff() + weight_w_rhs * get_rhs_weights_MS()
def loss_fit_test(): return fit_test() + weight_max * fit_test_max() + weight_min * fit_test_min() + weight_diff * fit_test_diff()

# Create and compile optimization problem.
opt = LNODEs.OptimizationProblem(rhs_base.variables + [z0_train], loss_fit_train, loss_fit_test)

# Training with Adam optimizer.
print('Training (Adam)...')
opt.optimize_keras(num_iters_ADAM, tf.keras.optimizers.Adam(learning_rate = 1e-2))
# Training with BFGS optimizer.
print('Training (BFGS)...')
opt.optimize_BFGS(num_iters_BFGS)

# Serialize rhs to JSON.
rhs_base_json = rhs_base.to_json()
with open(output_folder + 'model.json', "w") as json_file:
    json_file.write(rhs_base_json)

# Serialize weights to HDF5.
print('Save LNODEs...')
rhs_base.save_weights(output_folder + 'model.h5')

# Compute statistics on the testing set.
output_test = forward_solution(data_ip_test, n_samples_test, state_0_test, n_steps, num_out, dt, rhs)
RMSE = tf.math.sqrt(tf.reduce_mean(tf.square(output_test - data_op_test)))
RMSE = RMSE.numpy()
print('RMSE: %f' % RMSE)

NRMSE = tf.math.sqrt(tf.reduce_mean(tf.square(output_test - data_op_test))) / (tf.math.reduce_max(data_op_test) - tf.math.reduce_min(data_op_test))
NRMSE = NRMSE.numpy()
print('NRMSE: %f' % NRMSE)

R_coeff = scipy.stats.pearsonr(np.reshape(output_test.numpy(), (-1,)), np.reshape(data_op_test.numpy(), (-1,)))[0]
print('Pearson coeff: %f' % R_coeff)

cos_similarity_coeff = scipy.spatial.distance.cosine(np.reshape(output_test.numpy(), (-1,)), np.reshape(data_op_test.numpy(), (-1,)))
print('Cosine dissimilarity: %f' % cos_similarity_coeff)

# Plots.
plt.loglog(opt.iterations_history, opt.loss_train_history, 'o-', label = 'Training loss')
plt.loglog(opt.iterations_history, opt.loss_test_history, 'o-', label = 'Testing loss')
plt.axvline(num_iters_ADAM)
plt.xlabel('Epochs'), plt.ylabel('Loss function')
plt.legend()
plt.show()