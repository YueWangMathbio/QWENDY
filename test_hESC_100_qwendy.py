"""
analyze the results of QWENDY method on hESC_100 data
"""

import numpy as np
from methods import Qwendy_data
import warnings
warnings.filterwarnings("ignore")

gene_names = [
    "GATA6", "NANOG", "T", "EOMES", "ID2", "PRDM1", "ID1", "ZNF516", "SHOX2", "TBX3",
    "GATA4", "OTX2", "HAND1", "NFIB", "SOX2", "MSX1", "KLF8", "GRHL2", "ZNF521", "BHLHE40",
    "TFAP2A", "GATA3", "ZEB2", "BNC2", "KAT7", "NFE2L3", "ZEB1", "FOXF1", "PITX2", "HOXB3",
    "ID4", "ZNF165", "TCF7L2", "POU5F1", "SP5", "ZIC3", "TRPS1", "ID3", "AEBP2", "CDX1",
    "TCF7L1", "MIER1", "KLF6", "MYCN", "SMAD2", "PLAGL1", "ZSCAN10", "ZNF652", "HOXB6", "MSX2",
    "SOX5", "ZFP36L2", "LEF1", "BCL11A", "HBP1", "IRX3", "MEIS2", "ZFX", "TERF1", "TOX",
    "FOXH1", "HESX1", "SOX11", "KLF10", "ZIC5", "BBX", "RNF138", "SATB1", "BAZ2B", "ELK3",
    "ZBTB2", "ETV5", "SP6", "ZFP42", "TULP4", "ZNF471", "ARID4B", "SNAI2", "ZNF483", "MAF",
    "ETV6", "ZKSCAN1", "ETV1", "TMF1", "HIF3A", "SALL2", "SHOX", "SALL1", "SMAD7", "ZNF587",
    "SOX17", "ZFP62", "SRF", "JUND", "TUB", "ZFP14", "ARID3A", "CEBPZ", "E2F4", "TCF7"
]
core_genes = ["GATA6", "NANOG", "EOMES", "SOX2", "SOX17", "SMAD2", "FOXH1", "GATA4", "POU5F1"]
core_num = len(core_genes)
gene_num = [-1] * core_num
for j in range(core_num):
    name = core_genes[j]
    for i in range(100):
        if name == gene_names[i]:
            gene_num[j] = i
data = []
for i in range(4):  
    temp = np.load(f'hESC_100/hESC_data{i}.npy') # expression data
    data.append(temp)
total_data = data[:4]

grn = Qwendy_data(total_data)

total = [[0.0, i] for i in range(100)]
for i in range(100):
    for j in range(100):
        if i == j:
            continue
        total[i][0] += np.abs(grn[i, j])

total.sort(reverse=True, key=lambda x:x[0])

rank = [-1] * core_num
for i in range(core_num):
    for j in range(100):
        if total[j][1] == gene_num[i]:
            rank[i] = j + 1

for i in range(core_num):
    print(core_genes[i], 'rank:', rank[i])
print('Average rank:', np.mean(rank))
