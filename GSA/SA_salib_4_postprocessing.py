#!/usr/bin/env python3
import numpy as np
import csv
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    #"font.serif": ["Palatino"],
    "font.size": 16})
import seaborn as sns
import sys, getopt
import os
import utils

data_path    = '../output/'
figures_path = '../output/'
pb_name      = 'params'
inputfile    = 'sobol'
outputfile   = 'output'

opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "ofile="])
for opt, arg in opts:
    if opt == '-h':
        print('SA_salib_4_postprocessing.py -i <inputfile> -o <outputfile>')
        print('Input file is a .csv with Sobol indices')
        print('Output file is a .pdf which contains the heatmap')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg

print ('Input file:', inputfile)
print ('Output file:', outputfile)

problem = utils.get_problem()

# Define discrete color map.
sns.set_palette("bright")
myColors = ((0.97, 0.95, 0.97, 1.0), (0.83, 0.83, 0.91, 1.0), (0.38, 0.64, 0.80, 1.0), (0.02, 0.42, 0.66, 1.0), (0.008, 0.23, 0.36, 1.0))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

# Get data from csv.
sobol_labels = ['S1', 'S1_conf', 'ST', 'ST_conf']
sobol_indices = dict()
for SI in sobol_labels:
    rows = []
    with open(data_path + inputfile + '_' + SI + '_paper.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = -1
        for row in csv_reader:
            if line_count == -1:
                pass
            else:
                row_floats = []
                for item in row:
                    row_floats.append(float(item))
                rows.append(row_floats)
            line_count += 1
        print('Processed lines:', line_count)

    sobol_indices[SI] = np.array(rows)

outputs = {SI: sobol_indices[SI] for SI in sobol_labels}
outputs['S1_lower'] = sobol_indices['S1'] - sobol_indices['S1_conf']
outputs['S1_upper'] = sobol_indices['S1'] + sobol_indices['S1_conf']
outputs['ST_lower'] = sobol_indices['ST'] - sobol_indices['ST_conf']
outputs['ST_upper'] = sobol_indices['ST'] + sobol_indices['ST_conf']

order_plots_atria      = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 37, 38, 39]
order_plots_ventricles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 26, 27, 35, 36, 41, 42]
order_plots_heart      = [30, 40]
order_plots_circadapt  = [31, 32, 33, 34]
order_plots_input      = order_plots_atria + order_plots_ventricles + order_plots_heart + order_plots_circadapt
order_plots_output     = range(len(utils.output_labels))

for output_key in outputs.keys():
    print(output_key)
    fig, ax = plt.subplots(figsize = (24, 17))

    reordered_outputs = np.stack([outputs[output_key][i, :] for i in order_plots_input], axis = 0)
    final_outputs     = reordered_outputs

    ax = sns.heatmap(final_outputs, annot = True, linewidth = 0.5, fmt = '.2f', cbar_kws = {"shrink": .8}, cmap = cmap, vmin = 0) #, vmin = 0, vmax = 1) # "crest"
    
    plt.title(r"Sobol indices (%s)" % output_key, fontsize = 30)

    output_labels = [utils.output_labels[i] for i in order_plots_output]
    ax.xaxis.tick_top()
    plt.xticks(np.arange(len(utils.output_labels)) + .5, labels = output_labels, rotation = 70)

    params_names = [utils.params_names[i] for i in order_plots_input]
    ax.yaxis.tick_left()
    plt.yticks(np.arange(problem['num_vars']) + .5, labels = params_names, rotation = 0)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize = 30)

    for y in [0, len(order_plots_atria),
              len(order_plots_atria) + len(order_plots_ventricles),
              len(order_plots_atria) + len(order_plots_ventricles) + len(order_plots_heart),
              len(order_plots_input)]:
        ax.axhline(y = y, color = 'k', linewidth = 1.0)
    for x in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
        ax.axvline(x = x, color = 'k', linewidth = 1.0)

    fig.tight_layout()

    fig.savefig(figures_path + outputfile + '_' + output_key + '.pdf')
