"""
measure the performance of QWENDY, LEQWENDY, TEQWENDY methods on SINC data
"""

from evaluation import auroc_auprc
import numpy as np
from methods import Qwendy_Kx, Leqwendy_Kx, Teqwendy_Kx
import warnings
warnings.filterwarnings("ignore")


filename = 'SINC/SINC_cov.npy' # covariance
total_K_data = np.load(filename)

filename = 'SINC/SINC_mean.npy' # mean level
total_x_data = np.load(filename)

filename = 'SINC/SINC_A.npy' # true GRN
total_A = np.load(filename)

group_count = 1000
res = np.zeros((3, 2))
for i in range(1000):
    A = total_A[i]
    K_data = total_K_data[i]
    x_data = total_x_data[i]
    grn = Qwendy_Kx(K_data, x_data)
    auroc, auprc = auroc_auprc(A, grn)
    res[0, 0] += auroc / group_count
    res[0, 1] += auprc / group_count
    grn = Leqwendy_Kx(K_data, x_data)
    auroc, auprc = auroc_auprc(A, grn)
    res[1, 0] += auroc / group_count
    res[1, 1] += auprc / group_count
    grn = Teqwendy_Kx(K_data, x_data)
    auroc, auprc = auroc_auprc(A, grn)
    res[2, 0] += auroc / group_count
    res[2, 1] += auprc / group_count
res = np.round(res, 4)

print(res)
"""
rows: QWENDY, LEQWENDY, TEQWENDY
columns: AUROC, AUPRC
[[0.5314 0.572 ]
 [0.4972 0.5185]
 [0.9587 0.9142]]
"""