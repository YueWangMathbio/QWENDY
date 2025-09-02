"""
This file explains how to use QWENDY method to calculate the gene regulatory
network from single-cell level gene expression data, measured at four time 
points after some general interventions, where the joint distribution of 
these four time points is unknown.
"""

# either download qwendy.py file 
# or install qwendy package through pip install qwendy
from qwendy import qwendy_data

import numpy as np

# input data, list of 4 numpy arrays, each with shape (n_cells, n_genes)
# n_cells for different arrays can be different
# for more than 4 arrays, only keep the first 4
data = np.load('example_data.npy')

# if just want the inferred GRN
B = qwendy_data(data)
print('\nInferred GRN: ')
print(np.round(B, decimals=4))

# if want to print inferred regulations based on thresholds and gene names
B = qwendy_data(data, print_res=True, threshold_upper=0.5, threshold_lower=-0.5, 
                gene_names=['SOX17', 'FOXH1', 'EOMES', 'SMAD4'])
# if the number of provided gene names is less than actual gene number, add gene names
# if the number of provided gene names is more than actual gene number, delete extra gene names

