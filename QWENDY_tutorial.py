"""
This file explains how to use QWENDY method to calculate the gene regulatory
network from single-cell level gene expression data, measured at four time 
points after some general interventions, where the joint distribution of 
these four time points is unknown.
"""

# the input should be a list of four matrices
# each matrix is the expression data at one time point, of size
# cell number * gene number
# for instance, the input can be a numpy array of size (4, cell_num, gene_num)

import numpy as np
from methods import Qwendy_data, Leqwendy_data, Teqwendy_data
import warnings
warnings.filterwarnings("ignore")

total_data = np.load('example_data.npy')
grn_qwendy = Qwendy_data(total_data)
grn_leqwendy = Leqwendy_data(total_data)
grn_teqwendy = Teqwendy_data(total_data)

print('QWENDY GRN: ')
print(grn_qwendy)
print()

print('LEQWENDY GRN: ')
print(grn_leqwendy)
print()

print('TEQWENDY GRN: ')
print(grn_teqwendy)
print()


"""
if you have raw single-cell RNA sequencing (scRNAseq) data:
use scanpy or other packages to extract expression data at each time point.
remove genes that only appear in a few cells, and cells that only a few genes
are measured.
replace each value x by log(1+x).
for each cell (row), normalize its sum, so that each cell has the same 
total expression level. 
"""