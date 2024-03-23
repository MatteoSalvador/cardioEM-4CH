#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
import csv
import time
import math
import pickle
import random

import scipy.optimize as spopt
from scipy import interpolate

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_array', True)
from diffrax import diffeqsolve, ODETerm, Euler, SaveAt
import jax.numpy as jnp
from jax.test_util import check_grads
from jax.random import PRNGKey
from jax.experimental.ode import odeint

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import print_summary

import sys
sys.path.append('../cardioEM-4CH')
from utilities import *

import tensorflow as tf
from tensorflow.keras.models import model_from_json

class PE:

    def __init__(self, adaptive_solver,
                       MAP_output, 
                       test_label,
                       FOM_target_idx,
                       inverse_UQ,
                       ANN_folder, dataset_folder,
                       dt_base,
                       params_est = dict(),
                       QoIs_weights = dict()):

        # Adaptive solver (odeint) vs. forward euler.
        self.adaptive_solver = adaptive_solver

        # Enable/disable MAP output.
        self.MAP_output = MAP_output

        # Test case label.
        self.test_label = test_label

        # Input (target) settings.
        self.FOM_target_idx = FOM_target_idx

        # Flag for inverse UQ.
        self.inverse_UQ = inverse_UQ

        # ANN settings + dataset.
        self.ANN_folder = ANN_folder
        self.dataset_folder = dataset_folder
        self.dt_base = dt_base

        # Import ANN.
        self.json_file = open(self.ANN_folder + 'model.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.ANN_base = model_from_json(self.loaded_model_json)
        self.ANN_base.load_weights(self.ANN_folder + "model.h5")

        # Define ANN weights and biases.
        self.weights = []
        self.biases = []
        for layer in self.ANN_base.layers:
            w = jnp.array(layer.get_weights()[0])
            b = jnp.array(layer.get_weights()[1])
            self.weights.append(w)
            self.biases.append(b)

        # Read dataset and parameters.
        self.input_file = open(self.dataset_folder + 'simulations.pkl', "rb")
        self.dataset = pickle.load(self.input_file)
        self.parameters = jnp.load(self.dataset_folder + 'parameters.npy')

        # Inputs.
        self.inp_min = jnp.min(self.dataset['inp'], axis = (0, 1))
        self.inp_max = jnp.max(self.dataset['inp'], axis = (0, 1))
        self.num_inp = len(self.inp_min)

        # Outputs.
        self.num_out = self.dataset['out'].shape[2]

        # States.
        self.num_states = self.ANN_base.layers[-1].output.shape[1]
        self.state_min = jnp.min(self.dataset['out'], axis = (0, 1))
        self.state_max = jnp.max(self.dataset['out'], axis = (0, 1))
        self.num_mute_vars = int(self.num_states - self.num_out)
        
        # Heartbeat period.
        self.THB = 0.854 # [s].

        # Initial conditions for targets defined by ANN-based numerical simulation.
        if self.MAP_output:
            print('Initial conditions from 3D-0D numerical simulation...')
        
        # Define labels for parameters.
        self.labels = []
        self.labels.append('PCa_b'), self.labels.append('trpnmax'), self.labels.append('Gncx_b')
        self.labels.append('Tref_V'), self.labels.append('perm50_V'), self.labels.append('nperm_V')
        self.labels.append('TRPN_n_V'), self.labels.append('dr_V'), self.labels.append('wfrac_V')
        self.labels.append('TOT_A_V'), self.labels.append('ktm_unblock'), self.labels.append('ca50_V')
        self.labels.append('mu_V'), self.labels.append('maxI_up'), self.labels.append('maxTrpn')
        self.labels.append('g_CaL'), self.labels.append('Tref_A'), self.labels.append('perm50_A')
        self.labels.append('nperm_A'), self.labels.append('TRPN_n_A'), self.labels.append('dr_A')
        self.labels.append('wfrac_A'), self.labels.append('TOT_A_A'), self.labels.append('phi')
        self.labels.append('ca50_A'), self.labels.append('mu_A'), self.labels.append('CV_ventricles')
        self.labels.append('k_FEC'), self.labels.append('CV_atria'), self.labels.append('k_BB')
        self.labels.append('AV_delay'), self.labels.append('Rsys'), self.labels.append('Rpulm')
        self.labels.append('Aol'), self.labels.append('kArt'), self.labels.append('a_ventricles')
        self.labels.append('bt_ventricles'), self.labels.append('a_atria'), self.labels.append('bf_atria')
        self.labels.append('bt_atria'), self.labels.append('k_peri'), self.labels.append('a_lvrv')
        self.labels.append('Tref_lvrv')
        self.labels = np.array(self.labels)
        self.num_params = len(self.labels)

        # Initial values for the parameters.
        self.params_0 = []
        if self.MAP_output:
            print('Parameters initialization...')
        for idx in range(self.num_params):
            self.params_0.append(float(random.uniform(np.min(self.parameters, axis = 0)[idx], np.max(self.parameters, axis = 0)[idx])))
        self.params_0 = jnp.array(self.params_0)

        # Parameters for MAP estimation.
        if isinstance(params_est, str):
            with open(params_est, mode = 'r', newline = '') as inputfile:
                params_est = json.loads(inputfile.read())
        self.estimate_params = []
        for idx in range(self.num_params):
            self.estimate_params.append(bool(params_est.get(self.labels[idx], 0)))
        self.estimate_params = np.array(self.estimate_params)

        # Target values for the parameters.
        self.params_target = self.parameters[self.FOM_target_idx, :]

        # Put correct initial parameters for non-estimated ones in 3D-0D simulations.
        for idx in range(self.num_params):
            if not self.estimate_params[idx]:
                self.params_0 = self.params_0.at[idx].set(self.params_target[idx])

        # Loss weights.
        if isinstance(QoIs_weights, str):
            with open(QoIs_weights, mode = 'r', newline = '') as inputfile:
                QoIs_weights = json.loads(inputfile.read())
        QoIs_weights_curr = QoIs_weights.get('LA', dict())
        self.weight_V_LA = float(QoIs_weights_curr.get('volume'  , 0.))
        self.weight_p_LA = float(QoIs_weights_curr.get('pressure', 0.))
        QoIs_weights_curr = QoIs_weights.get('LV', dict())
        self.weight_V_LV = float(QoIs_weights_curr.get('volume'  , 0.))
        self.weight_p_LV = float(QoIs_weights_curr.get('pressure', 0.))
        QoIs_weights_curr = QoIs_weights.get('RA', dict())
        self.weight_V_RA = float(QoIs_weights_curr.get('volume'  , 0.))
        self.weight_p_RA = float(QoIs_weights_curr.get('pressure', 0.))
        QoIs_weights_curr = QoIs_weights.get('RV', dict())
        self.weight_V_RV = float(QoIs_weights_curr.get('volume'  , 0.))
        self.weight_p_RV = float(QoIs_weights_curr.get('pressure', 0.))

        # Define right hand sides.
        self.rhs_base = lambda x, u: self.ANN(jnp.concatenate([x, u], axis = 0), self.weights, self.biases)
        self.rhs = lambda state, inputs: redim(self.rhs_base(adim(state, self.state_min, self.state_max), adim(inputs, self.inp_min, self.inp_max)), self.state_min, self.state_max) / self.dt_base
        if self.adaptive_solver:
            self.rhs_odeint = lambda state, t, params: self.rhs(state, jnp.concatenate([jnp.array([jnp.cos(2 * jnp.pi * (t - (params[30] / 1000.)) / self.THB), jnp.sin(2 * jnp.pi * (t - (params[30] / 1000.)) / self.THB)]), params], axis = 0))
        else:
            self.rhs_odeint = lambda t, state, params: self.rhs(state, jnp.concatenate([jnp.array([jnp.cos(2 * jnp.pi * (t - (params[30] / 1000.)) / self.THB), jnp.sin(2 * jnp.pi * (t - (params[30] / 1000.)) / self.THB)]), params], axis = 0))
            # Diffrax initialization (forward euler method).
            self.rhs_diffrax = ODETerm(self.rhs_odeint)
            self.solver = Euler()

        # Discrete L^2 relative errors.
        self.l2_errors = []

    def ANN(self, input, weights, biases):
        y = input
        for i in range(len(weights)):
            y = jnp.matmul(jnp.transpose(weights[i]), y) + biases[i]
            if i < len(weights) - 1:
                y = jnp.tanh(y)
        return y

    def solve(self, x, params):
        if self.adaptive_solver:
            y = odeint(self.rhs_odeint, x, self.times, params)
            y = y[-self.last_cycle_init:, :]
        else:
            sol = diffeqsolve(self.rhs_diffrax, self.solver, t0 = self.times[0], t1 = self.times[-1], dt0 = self.dt, y0 = x, args = params, saveat = self.saveat)
            y = sol.ys[-self.last_cycle_init:, :]

        return y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6], y[:, 7]

    def loss(self, params):
        self.p_LA, self.p_LV, self.p_RA, self.p_RV, self.V_LA, self.V_LV, self.V_RA, self.V_RV = self.solve(self.initial_state, params)

        losses = list()

        # p_LA.
        losses.append(0.0 if self.weight_p_LA < 1e-15 else self.weight_p_LA * jnp.mean(jnp.square(self.p_LA - self.p_LA_opt)) / self.normalization_p_LA)
        # p_LV.
        losses.append(0.0 if self.weight_p_LV < 1e-15 else self.weight_p_LV * jnp.mean(jnp.square(self.p_LV - self.p_LV_opt)) / self.normalization_p_LV)
        # p_RA.
        losses.append(0.0 if self.weight_p_RA < 1e-15 else self.weight_p_RA * jnp.mean(jnp.square(self.p_RA - self.p_RA_opt)) / self.normalization_p_RA)
        # p_RV.
        losses.append(0.0 if self.weight_p_RV < 1e-15 else self.weight_p_RV * jnp.mean(jnp.square(self.p_RV - self.p_RV_opt)) / self.normalization_p_RV)        
        # V_LA.
        losses.append(0.0 if self.weight_V_LA < 1e-15 else self.weight_V_LA * jnp.mean(jnp.square(self.V_LA - self.V_LA_opt)) / self.normalization_V_LA)
        # V_LV.
        losses.append(0.0 if self.weight_V_LV < 1e-15 else self.weight_V_LV * jnp.mean(jnp.square(self.V_LV - self.V_LV_opt)) / self.normalization_V_LV)
        # V_RA.
        losses.append(0.0 if self.weight_V_RA < 1e-15 else self.weight_V_RA * jnp.mean(jnp.square(self.V_RA - self.V_RA_opt)) / self.normalization_V_RA)
        # V_RV.
        losses.append(0.0 if self.weight_V_RV < 1e-15 else self.weight_V_RV * jnp.mean(jnp.square(self.V_RV - self.V_RV_opt)) / self.normalization_V_RV)

        return sum(losses)        

    def func_grad(self, params):
        t_init = time.time()
        l = float(np.array(self.loss_jit(params)))
        g = np.array(self.grad_jit(params))
        if self.MAP_output:
            print('loss: %1.16e (%f s)' % (l, time.time() - t_init))
        return l, g

    def func(self, params):
        l = float(np.array(self.loss_jit(params)))
        if self.MAP_output:
            print('loss: %1.16e' % l)
        return l

    def target(self, path_to_csv, x_label, y_label, t_min, t_max, times):
        history = pd.read_csv(path_to_csv, header = 0)
        history = history.reset_index(drop = True)
        history = history.rename(columns = lambda x: x.strip())
        history = history.drop(0)
        history[x_label] = pd.to_numeric(history[x_label], downcast = "float")
        history[y_label] = pd.to_numeric(history[y_label], downcast = "float")
        history = history[history[x_label] >= t_min]
        history = history[history[x_label] <= t_max]
        tck = interpolate.splrep([x for x in history[x_label]], history[y_label])
        target = interpolate.splev(times, tck)
        target_vec = np.asarray(target)

        return target

    def run(self, num_cycles, dt, output_folder = 'output', filename_output = 'circulation_MAP'):

        # Time settings.
        self.num_cycles = num_cycles
        self.dt = dt
        self.times = np.arange(0.0, (self.num_cycles * self.THB) + 1e-4, self.dt)
        self.nT = self.times.shape[0]
        self.last_cycle_init = int(self.nT / self.num_cycles)
        self.times_last_cycle = self.times[-self.last_cycle_init:]
        if not self.adaptive_solver:
            self.saveat = SaveAt(ts = self.times)

        # Output settings.
        self.filename_output = filename_output
        self.output_folder = output_folder

        if self.MAP_output:
            print('Get target(s) from 3D-0D numerical simulation...')
        self.dataset['times'] = self.dataset['times'] / 1000. # [ms] -> [s].
        self.last_HB_target = int(3 * 1000 * self.THB - 1)
        self.times_target = np.arange(self.dataset['times'][self.last_HB_target], self.dataset['times'][-1] + self.dt * 1e-10, step = self.dt)
        self.FOM_target_traces = interpolate.interp1d(self.dataset['times'], self.dataset['out'][self.FOM_target_idx, :, :], axis = 0)(self.times_target)
        self.p_LA_opt = self.FOM_target_traces[:, 0]
        self.p_LV_opt = self.FOM_target_traces[:, 1]
        self.p_RA_opt = self.FOM_target_traces[:, 2]
        self.p_RV_opt = self.FOM_target_traces[:, 3]
        self.V_LA_opt = self.FOM_target_traces[:, 4]
        self.V_LV_opt = self.FOM_target_traces[:, 5]
        self.V_RA_opt = self.FOM_target_traces[:, 6]
        self.V_RV_opt = self.FOM_target_traces[:, 7]

        # Initial conditions are fixed from 3D-0D numerical simulations.
        self.initial_state = []
        self.initial_state.append(self.p_LA_opt[0])
        self.initial_state.append(self.p_LV_opt[0])
        self.initial_state.append(self.p_RA_opt[0])
        self.initial_state.append(self.p_RV_opt[0])
        self.initial_state.append(self.V_LA_opt[0])
        self.initial_state.append(self.V_LV_opt[0])
        self.initial_state.append(self.V_RA_opt[0])
        self.initial_state.append(self.V_RV_opt[0])
        self.initial_state = jnp.array(self.initial_state)

        # Compute normalization factors.
        if self.weight_p_LA >= 1e-15:
            self.normalization_p_LA = jnp.mean(jnp.square(self.p_LA_opt))
        if self.weight_p_LV >= 1e-15:
            self.normalization_p_LV = jnp.mean(jnp.square(self.p_LV_opt))
        if self.weight_p_RA >= 1e-15:            
            self.normalization_p_RA = jnp.mean(jnp.square(self.p_RA_opt))
        if self.weight_p_RV >= 1e-15:            
            self.normalization_p_RV = jnp.mean(jnp.square(self.p_RV_opt))
        if self.weight_V_LA >= 1e-15:
            self.normalization_V_LA = jnp.mean(jnp.square(self.V_LA_opt))
        if self.weight_V_LV >= 1e-15:
            self.normalization_V_LV = jnp.mean(jnp.square(self.V_LV_opt))
        if self.weight_V_RA >= 1e-15:            
            self.normalization_V_RA = jnp.mean(jnp.square(self.V_RA_opt))
        if self.weight_V_RV >= 1e-15:            
            self.normalization_V_RV = jnp.mean(jnp.square(self.V_RV_opt))

        # Define loss and gradients + just-in-time compilation.
        self.grad = jax.grad(self.loss)
        self.loss_jit = jax.jit(self.loss)
        self.grad_jit = jax.jit(self.grad)
        if self.MAP_output:
            print('Loss function compilation...')
        t_init = time.time()
        l,g = self.func_grad(self.params_0)
        if self.MAP_output:
            print('Compiled (%f s)' % (time.time() - t_init))

        # Parameters bounds.
        self.define_bounds()

        # Define parameters labels for MAP estimation.
        self.labels_MAP = []
        for idx in range(self.num_params):
            if self.estimate_params[idx]:
                self.labels_MAP.append(self.labels[idx])

        # Callback of the optimizer (print parameters values). 
        self.iteration = 0
        def callback(params):
            print('====================================== iteration %d' % self.iteration)

            for idx in range(self.num_params):
                if self.estimate_params[idx]:
                    print(self.labels[idx] + ': %1.16f / %1.16f' % (params[idx], self.params_target[idx]))

            self.iteration += 1

            self.compute_errors(params)
            print('**************************************')
            print('L2 error: %1.16f' % self.l2_errors[-1])
                
            return True

        # Optimization with L-BFGS.
        print('******* MAP estimation with jax (simulation = ' + str(self.FOM_target_idx) + ', test case = ' + self.test_label + ') *******', flush = True)
        self.max_iter = 100000
        t_init = time.time()
        if self.MAP_output:
            ret = spopt.minimize(fun = self.func_grad, x0 = self.params_0, jac = True,
                                 method = 'L-BFGS-B', bounds = self.bounds,
                                 options = {'ftol': 1e-100, 'gtol': 1e-100, 'maxiter': self.max_iter},
                                 callback = callback)
        else:
            ret = spopt.minimize(fun = self.func_grad, x0 = self.params_0, jac = True,
                                 method = 'L-BFGS-B', bounds = self.bounds,
                                 options = {'ftol': 1e-100, 'gtol': 1e-100, 'maxiter': self.max_iter})
        opt_time = time.time() - t_init
        print('Optimization time: %f s' % opt_time)
        self.params_MAP = ret['x'].tolist()
        loss_opt, grad_opt = self.func_grad(ret['x'])

        # Get estimated parameters only.
        self.params_MAP_estimated = []
        for idx in range(self.num_params):
            if self.estimate_params[idx]:
                self.params_MAP_estimated.append(self.params_MAP[idx])

        # Compute the final error for FOM target only, when MAP output is disabled.
        if not self.MAP_output:
            self.compute_errors(self.params_MAP) 

        # Output MAP results.
        if self.MAP_output:
            print(ret['message'])
            print('nit: %d' % ret['nit'])
            print('Parameters (target):')
            print(self.params_target)
            print('Discrete L^2 relative error:')
            print(self.l2_errors[-1])

        if self.inverse_UQ:
            print('******* Inverse UQ with numpyro *******')
            
            # HMC settings.
            self.num_samples = 500
            self.num_chains = 1
            self.num_warmup = 250
            self.device = 'cpu'

            # Surrogate modeling errors.
            self.ROM_error_p_LA = GP_generator(jnp.array(self.times_last_cycle), 0., 0.13, 0.02).get_sigma()
            self.ROM_error_p_LV = GP_generator(jnp.array(self.times_last_cycle), 0., 2.30, 0.02).get_sigma()
            self.ROM_error_p_RA = GP_generator(jnp.array(self.times_last_cycle), 0., 0.09, 0.02).get_sigma()
            self.ROM_error_p_RV = GP_generator(jnp.array(self.times_last_cycle), 0., 0.46, 0.02).get_sigma()
            self.ROM_error_V_LA = GP_generator(jnp.array(self.times_last_cycle), 0., 1.82, 0.02).get_sigma()
            self.ROM_error_V_LV = GP_generator(jnp.array(self.times_last_cycle), 0., 1.34, 0.02).get_sigma()
            self.ROM_error_V_RA = GP_generator(jnp.array(self.times_last_cycle), 0., 2.29, 0.02).get_sigma()
            self.ROM_error_V_RV = GP_generator(jnp.array(self.times_last_cycle), 0., 1.38, 0.02).get_sigma()

            # Measurement errors.
            self.zeros_mat = jnp.zeros((self.last_cycle_init, self.last_cycle_init))
            diag_elements = jnp.diag_indices_from(self.zeros_mat)
            self.meas_error_p_LA = self.zeros_mat.at[diag_elements].set(0.0 + 0.13 * 0.001)
            self.meas_error_p_LV = self.zeros_mat.at[diag_elements].set(0.0 + 2.30 * 0.001)
            self.meas_error_p_RA = self.zeros_mat.at[diag_elements].set(0.0 + 0.09 * 0.001)
            self.meas_error_p_RV = self.zeros_mat.at[diag_elements].set(0.0 + 0.46 * 0.001)
            self.meas_error_V_LA = self.zeros_mat.at[diag_elements].set(0.0 + 1.82 * 0.001)
            self.meas_error_V_LV = self.zeros_mat.at[diag_elements].set(0.0 + 1.34 * 0.001)
            self.meas_error_V_RA = self.zeros_mat.at[diag_elements].set(0.0 + 2.29 * 0.001)
            self.meas_error_V_RV = self.zeros_mat.at[diag_elements].set(0.0 + 1.38 * 0.001)
            
            # Initialize HMC.            
            dict_init_value = {}
            for idx in range(self.num_params):
                if self.estimate_params[idx]:
                    dict_init_value[self.labels[idx]] = self.params_MAP[idx]

            init_strategy = numpyro.infer.init_to_value(values = dict_init_value)
            kernel = NUTS(self.define_distributions, step_size = 1e-3, dense_mass = True, find_heuristic_step_size = False, max_tree_depth = 5, init_strategy = init_strategy)
            mcmc = MCMC(kernel, num_warmup = self.num_warmup, num_samples = self.num_samples, num_chains = self.num_chains, progress_bar = True)

            # Run HMC.
            mcmc.run(PRNGKey(1))

            # Print the results related to all posterior parameters.
            mcmc.print_summary()

            mcmc_samples = mcmc.get_samples()
            
            # Generate predictions from the posterior distribution.
            print('Generating predictions... ')

            params_pred = []
            dict_params_pred = {}
            for idx in range(self.num_params):
                if self.estimate_params[idx]:
                    dict_params_pred[self.labels[idx]] = mcmc_samples[self.labels[idx]].tolist()

            self.params_pred_df = pd.DataFrame(dict_params_pred)
            self.params_pred_df.to_csv(self.output_folder + '/statistics_HMC_' + self.test_label + '_' + str(self.FOM_target_idx) + '.csv', index = False)

        return opt_time, self.l2_errors[-1], ret['nit'], loss_opt, self.params_MAP_estimated, self.labels_MAP 

    def define_bounds(self):
        self.n_est_params = 0
        self.bounds = []
        
        for idx in range(self.num_params):
            if self.estimate_params[idx]:
                self.bounds += [[np.min(self.parameters, axis = 0)[idx], np.max(self.parameters, axis = 0)[idx]]]
                self.n_est_params += 1
            else:
                self.bounds += [[self.params_0[idx], self.params_0[idx]]]

    def define_distributions(self):
            self.distributions = []
            self.factor = 0.1

            # Define prior distributions in the parameter space.
            for idx in range(self.num_params):
                if self.estimate_params[idx]:
                    low_factor = self.params_MAP[idx] - self.factor * self.params_MAP[idx]
                    high_factor = self.params_MAP[idx] + self.factor * self.params_MAP[idx]
                    if low_factor > float(self.bounds[idx][0]):
                        low_factor = float(self.bounds[idx][0])
                    if high_factor < float(self.bounds[idx][1]):
                        high_factor = float(self.bounds[idx][1])
                    self.distributions.append(numpyro.sample(self.labels[idx], dist.Uniform(low  = low_factor ,
                                                                                            high = high_factor)))
                else:
                    self.distributions.append(self.params_MAP[idx])
            self.distributions = jnp.array(self.distributions)

            # Solve forward model.
            self.p_LA, self.p_LV, self.p_RA, self.p_RV, self.V_LA, self.V_LV, self.V_RA, self.V_RV = self.solve(self.initial_state, self.distributions)

            # ROM error(s) + measurement error on selected QoIs.
            if self.weight_p_LA >= 1e-15:
                numpyro.sample("p_LA", dist.MultivariateNormal(loc = self.p_LA, covariance_matrix = self.ROM_error_p_LA + self.meas_error_p_LA), obs = self.p_LA_opt)
            if self.weight_p_LV >= 1e-15:
                numpyro.sample("p_LV", dist.MultivariateNormal(loc = self.p_LV, covariance_matrix = self.ROM_error_p_LV + self.meas_error_p_LV), obs = self.p_LV_opt)
            if self.weight_p_RA >= 1e-15:                
                numpyro.sample("p_RA", dist.MultivariateNormal(loc = self.p_RA, covariance_matrix = self.ROM_error_p_RA + self.meas_error_p_RA), obs = self.p_RA_opt)                
            if self.weight_p_RV >= 1e-15:               
                numpyro.sample("p_RV", dist.MultivariateNormal(loc = self.p_RV, covariance_matrix = self.ROM_error_p_RV + self.meas_error_p_RV), obs = self.p_RV_opt)                
            if self.weight_V_LA >= 1e-15:
                numpyro.sample("V_LA", dist.MultivariateNormal(loc = self.V_LA, covariance_matrix = self.ROM_error_V_LA + self.meas_error_V_LA), obs = self.V_LA_opt)
            if self.weight_V_LV >= 1e-15:
                numpyro.sample("V_LV", dist.MultivariateNormal(loc = self.V_LV, covariance_matrix = self.ROM_error_V_LV + self.meas_error_V_LV), obs = self.V_LV_opt)
            if self.weight_V_RA >= 1e-15:               
                numpyro.sample("V_RA", dist.MultivariateNormal(loc = self.V_RA, covariance_matrix = self.ROM_error_V_RA + self.meas_error_V_RA), obs = self.V_RA_opt)                
            if self.weight_V_RV >= 1e-15:
                numpyro.sample("V_RV", dist.MultivariateNormal(loc = self.V_RV, covariance_matrix = self.ROM_error_V_RV + self.meas_error_V_RV), obs = self.V_RV_opt)

    def compute_errors(self, params):
        l2_error = 0.
        idx = 0
        for t, o in zip(self.params_target, params):
            if self.estimate_params[idx]:
                l2_error += ((t - o) / t) ** 2
            idx = idx + 1
        l2_error = l2_error / self.n_est_params
        l2_error = math.sqrt(l2_error)
        self.l2_errors.append(l2_error)