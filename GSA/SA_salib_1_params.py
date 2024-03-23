#!/usr/bin/env python3
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import sys, getopt
import utils

data_folder = '../data/'
nsamples = 8000
outputfile = "param_values.csv"

opts, args = getopt.getopt(sys.argv[1:], "h:n:o:", ["nsamples=", "ofile="])
for opt, arg in opts:
    if opt == '-h':
        print('SA_salib_1_params.py -n <numberofsamples> -o <outputfile>')
        print('Number of samples to generate (integer greater than 0)')
        print('Output file is a .csv which contains the samples')
        sys.exit()
    elif opt in ("-n", "--nsamples"):
        nsamples = int(arg)
    elif opt in ("-o", "--ofile"):
        outputfile = arg
print ('Number of samples:', str(nsamples))
print ('Output file:', outputfile)

problem = utils.get_problem()

param_values = saltelli.sample(problem, nsamples, calc_second_order = True)
np.savetxt(data_folder + outputfile, param_values, delimiter = ",")
