"""
measure the performance of QWENDY, LEQWENDY, TEQWENDY methods on THP-1 data
"""

from evaluation import auroc_auprc
import numpy as np
from methods import Qwendy_data, Leqwendy_data, Teqwendy_data
import warnings
warnings.filterwarnings("ignore")

all_time = [0, 1, 6, 12, 24, 48, 72, 96]
A = np.load('THP1/THP1_A.npy') # true GRN
data = np.load('THP1/THP1_data.npy') # expression data

res = np.zeros((3, 2))

total_data = [[data[0], data[4], data[5], data[6]], [data[4], data[5], data[6], data[7]]]
for i in range(2):
    grn = Qwendy_data(total_data[i])
    auroc, auprc = auroc_auprc(A, grn)
    res[0, 0] += auroc / 2
    res[0, 1] += auprc / 2
    grn = Leqwendy_data(total_data[i])
    auroc, auprc = auroc_auprc(A, grn)
    res[1, 0] += auroc / 2
    res[1, 1] += auprc / 2
    grn = Teqwendy_data(total_data[i])
    auroc, auprc = auroc_auprc(A, grn)
    res[2, 0] += auroc / 2
    res[2, 1] += auprc / 2
res = np.round(res, 4)

print(res)
"""
rows: QWENDY, LEQWENDY, TEQWENDY
columns: AUROC, AUPRC
[[0.5524 0.4294]
 [0.5543 0.3632]
 [0.5415 0.3801]]
"""

