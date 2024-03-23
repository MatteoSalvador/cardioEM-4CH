#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
import json
import os
import sys, getopt
import copy
import pickle

import tensorflow as tf
from tensorflow.keras.models import model_from_json

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Euler, SaveAt

import utils

sys.path.insert(1, '..')

sys.path.append('../cardioEM-4CH')
from utilities import *

@jax.jit
def ANN(input, weights, biases):
    y = input
    for i in range(len(weights)):
        y = jnp.matmul(jnp.transpose(weights[i]), y) + biases[i]
        if i < len(weights) - 1:
            y = jnp.tanh(y)
    return y

@jax.jit
def solve(x, params):
    sol = diffeqsolve(rhs_diffrax, solver, t0 = times_last_cycle[0], t1 = times_last_cycle[-1], dt0 = dt, y0 = x, args = params, saveat = saveat)
    y = sol.ys

    return y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6], y[:, 7]

# Parameters (sampling).
data_folder = '../data/'
inputfile  = "param_values.csv"
outputfile = "qoi.csv"

# Parameters (LNODE).
ANN_folder = '../ANNs/rhs_13neur_3lay_8states_0.023L2reg_0.0285dtbase_paper/'

# Parameters (times).
THB        = 0.854
num_cycles = 4
dt         = 1e-3 # [s].
dt_base    = 0.0285 # [s].

# Setup.
opts, args = getopt.getopt(sys.argv[1:], "h:i:o:", ["ifile=", "ofile="])
for opt, arg in opts:
    if opt == '-h':
        print('SA_salib_2_execution.py -i <inputfile> -o <outputfile>')
        print('Input file is a .csv which contains the samples')
        print('Output file is a .csv which contains the qoi')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg

print ('Input file:', inputfile)

if os.path.exists(data_folder + outputfile):
  os.remove(data_folder + outputfile)

param_values = np.genfromtxt(data_folder + inputfile, delimiter = ',')

json_file = open(ANN_folder + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ANN_base = model_from_json(loaded_model_json)
ANN_base.load_weights(ANN_folder + "model.h5")

weights = []
biases  = []
for layer in ANN_base.layers:
    w = jnp.array(layer.get_weights()[0])
    b = jnp.array(layer.get_weights()[1])
    weights.append(w)
    biases.append(b)

input_file = open(data_folder + 'simulations.pkl', "rb")
dataset    = pickle.load(input_file)

inp_min = jnp.min(dataset['inp'], axis = (0, 1))
inp_max = jnp.max(dataset['inp'], axis = (0, 1))
num_inp = len(inp_min)

num_out = dataset['out'].shape[2]

num_states = ANN_base.layers[-1].output.shape[1]
state_min  = jnp.min(dataset['out'], axis = (0, 1))
state_max  = jnp.max(dataset['out'], axis = (0, 1))

rhs_base    = lambda x, u: ANN(jnp.concatenate([x, u], axis = 0), weights, biases)
rhs         = lambda state, inputs: redim(rhs_base(adim(state, state_min, state_max), adim(inputs, inp_min, inp_max)), state_min, state_max) / dt_base
rhs_odeint  = lambda t, state, params: rhs(state, jnp.concatenate([jnp.array([jnp.cos(2 * jnp.pi * (t - (params[30] / 1000.)) / THB), jnp.sin(2 * jnp.pi * (t - (params[30] / 1000.)) / THB)]), params], axis = 0))
rhs_diffrax = ODETerm(rhs_odeint)

solver = Euler()

times            = jnp.arange(0.0, (num_cycles * THB) + 1e-4, dt)
nT               = times.shape[0]
last_cycle_init  = int(nT / num_cycles)
times_last_cycle = times[-last_cycle_init:]
saveat           = SaveAt(ts = times_last_cycle)

# Loop over samples.
tt = time.time()
for current_row in range(param_values.shape[0]):
    initial_state = np.array([4.75, 5.32, 2.78, 3.25, 158.84, 141.37, 137.38, 124.07])
    params_curr = copy.deepcopy(param_values[current_row, :43])

    tic = time.time()
    p_LA, p_LV, p_RA, p_RV, V_LA, V_LV, V_RA, V_RV = solve(initial_state, params_curr)
    elapsed = time.time() - tic
    print('Solution time: %s seconds' % (elapsed))

    QoI  = np.array([np.min(p_LA), np.max(p_LA), np.min((p_LA[1:] - p_LA[:-1]) / dt), np.max((p_LA[1:] - p_LA[:-1]) / dt),
                     np.min(V_LA), np.max(V_LA), np.min((V_LA[1:] - V_LA[:-1]) / dt), np.max((V_LA[1:] - V_LA[:-1]) / dt),
                     np.min(p_LV), np.max(p_LV), np.min((p_LV[1:] - p_LV[:-1]) / dt), np.max((p_LV[1:] - p_LV[:-1]) / dt),
                     np.min(V_LV), np.max(V_LV), np.min((V_LV[1:] - V_LV[:-1]) / dt), np.max((V_LV[1:] - V_LV[:-1]) / dt),
                     np.min(p_RA), np.max(p_RA), np.min((p_RA[1:] - p_RA[:-1]) / dt), np.max((p_RA[1:] - p_RA[:-1]) / dt),
                     np.min(V_RA), np.max(V_RA), np.min((V_RA[1:] - V_RA[:-1]) / dt), np.max((V_RA[1:] - V_RA[:-1]) / dt),
                     np.min(p_RV), np.max(p_RV), np.min((p_RV[1:] - p_RV[:-1]) / dt), np.max((p_RV[1:] - p_RV[:-1]) / dt),
                     np.min(V_RV), np.max(V_RV), np.min((V_RV[1:] - V_RV[:-1]) / dt), np.max((V_RV[1:] - V_RV[:-1]) / dt)])

    with open(data_folder + outputfile, "ab") as f:
        np.savetxt(f, [QoI], delimiter = ',')

    #print('Done row %d' % (current_row))

elapsed = time.time() - tt
print('Elapsed time: %s seconds' % (elapsed))
