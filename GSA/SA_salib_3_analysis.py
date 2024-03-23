#!/usr/bin/env python3
from SALib.analyze import sobol
import numpy as np
import sys, getopt
import csv
import utils

data_folder = '../data/'
output_folder = '../output/'
inputfile = "qoi.csv"
outputfile = 'sobol'
sobol_index = 'All'

opts, args = getopt.getopt(sys.argv[1:], "h:i:o:s:", ["ifile=", "ofile=", "sfile="])
for opt, arg in opts:
    if opt == '-h':
        print('SA_salib_3_analysis.py -i <inputfile> -o <outputfile> -s <sobolindices>')
        print('Input file is a .csv which contains the qoi')
        print('Output file is a .csv which contains Sobol indices')
        print('Sobol indices to export. Available options are: S1|S1_conf|ST|ST_conf|All')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt in ("-s", "--sfile"):
        sobol_index = arg

print ('Output file:', outputfile)
print ('Sobol indices:', sobol_index)

problem = utils.get_problem()

outputs = np.genfromtxt(data_folder + inputfile, delimiter = ',')

output_matrix = np.zeros((problem['num_vars'], len(utils.output_labels)))

if sobol_index == 'All':
    sobol_indices = ['S1', 'S1_conf', 'ST', 'ST_conf']
    files_output  = [output_folder + outputfile + '_' + SI + '.csv' for SI in sobol_indices]
else:
    sobol_indices = [sobol_index]
    files_output  = [output_folder + outputfile]

output_matrix = {SI: np.zeros((problem['num_vars'], len(utils.output_labels))) for SI in sobol_indices}

for QoI_index in range(len(utils.output_labels)):
    Si = sobol.analyze(problem, outputs[:, QoI_index], calc_second_order = True, print_to_console = False)
    for SI in sobol_indices:
        output_matrix[SI][:, QoI_index] = Si[SI]

for SI, file_out in zip(sobol_indices, files_output):
    with open(file_out, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(utils.output_labels)
        for index in range(problem['num_vars']):
            writer.writerow(output_matrix[SI][index, :])
