"""
measure the performance of QWENDY, LEQWENDY, TEQWENDY methods on DREAM4 data
"""

from evaluation import auroc_auprc
import numpy as np
from methods import Qwendy_data, Leqwendy_data, Teqwendy_data
import warnings
warnings.filterwarnings("ignore")

all_time = list(range(0, 1050, 50))

res = np.zeros((3, 2))
group_count = 90
for network in range(5):
    A = np.load(f'DREAM4/DREAM4_A_10_{network}.npy') # true GRN
    data = np.load(f'DREAM4/DREAM4_data_10_{network}.npy') # expression data
    for st in range(18):
        total_data = data[st: st+4]
        grn = Qwendy_data(total_data)
        auroc, auprc = auroc_auprc(A, grn)
        res[0, 0] += auroc / group_count
        res[0, 1] += auprc / group_count
        grn = Leqwendy_data(total_data)
        auroc, auprc = auroc_auprc(A, grn)
        res[1, 0] += auroc / group_count
        res[1, 1] += auprc / group_count
        grn = Teqwendy_data(total_data)
        auroc, auprc = auroc_auprc(A, grn)
        res[2, 0] += auroc / group_count
        res[2, 1] += auprc / group_count
res = np.round(res, 4)
    
print(res) 

"""
rows: QWENDY, LEQWENDY, TEQWENDY
columns: AUROC, AUPRC
[[0.4987 0.1844]
 [0.5164 0.1823]
 [0.5372 0.2203]]
"""
