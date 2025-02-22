"""
measure the performance of QWENDY, LEQWENDY, TEQWENDY methods on hESC data
"""

from evaluation import auroc_auprc
import numpy as np
from methods import Qwendy_data, Leqwendy_data, Teqwendy_data
import warnings
warnings.filterwarnings("ignore")

all_time = [0, 12, 24, 36, 72, 96]
data = []
for i in range(6):   
    temp = np.load(f'hESC/hESC_data{i}.npy') # expression data
    data.append(temp)
A = np.load('hESC/hESC_A.npy') # true GRN

res = np.zeros((3, 2))

total_data = data[:4]

grn = Qwendy_data(total_data)
auroc, auprc = auroc_auprc(A, grn)
res[0, 0] += auroc 
res[0, 1] += auprc 
grn = Leqwendy_data(total_data)
auroc, auprc = auroc_auprc(A, grn)
res[1, 0] += auroc 
res[1, 1] += auprc 
grn = Teqwendy_data(total_data)
auroc, auprc = auroc_auprc(A, grn)
res[2, 0] += auroc 
res[2, 1] += auprc 
res = np.round(res, 4)

print(res)
"""
rows: QWENDY, LEQWENDY, TEQWENDY
columns: AUROC, AUPRC
[[0.6019 0.0435]
 [0.5905 0.0367]
 [0.4815 0.0317]]
"""


