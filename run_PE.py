#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import PE

# Label for estimation, weights and outputs, according to the specific test case.
label = 'TLV' # ['TLV', 'Tatria', 'Tventricles', 'Tall'].

# If this flag is true, HMC runs after MAP estimation.
inverse_UQ = True

# Parameters for MAP estimation and HMC.
params_estimation_file = 'params/estimation_' + label + '.json'

# Target values for the parameters (3D-0D numerical simulation).
FOM_simulations = [262, 365, 230, 10, 33]

# Loss function weights.
QoIs_weights_file = 'QoIs/weights_' + label + '.json'

# Output.
output_folder = 'output'
filename_output = '4CH-ANN'

# ANN folder.
ANN_folder = './ANNs/rhs_13neur_3lay_8states_0.023L2reg_0.0285dtbase_paper/'
# ANN base timestep.
dt_base = 0.0285 # [s].
# Dataset folder.
dataset_folder = './data/'

for idx_FOM_simulation in range(len(FOM_simulations)):
	# Initialization.
	par_est = PE.PE(adaptive_solver = False,
					MAP_output = True,
					test_label = label,
					FOM_target_idx = FOM_simulations[idx_FOM_simulation],
					inverse_UQ = inverse_UQ,
				    ANN_folder = ANN_folder,
					dataset_folder = dataset_folder,
					dt_base = dt_base,
				    params_est = params_estimation_file,
				    QoIs_weights = QoIs_weights_file)

	# Run MAP and HMC for parameter estimation.
	par_est.run(num_cycles = 1, # One heartbeat.
				dt = 1e-3, # [s].
		        output_folder = output_folder,
		        filename_output = filename_output)