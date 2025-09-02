# QWENDY

code files for QWENDY, LEQWENDY, TEQWENDY methods, used for inferring gene regulatory networks (GRN) from single-cell gene expression data

to use QWENDY, either use 
pip install qwendy
in terminal, or directly download the qwendy.py file

the paper is on arXiv: https://arxiv.org/abs/2503.09605

QWENDY algorithm uses numpy=1.24.3, sklearn=1.4.2

qwendy.py: main function of the QWENDY method

see qwendy_tutorial_new.py for a simple tutorial to use QWENDY


TEQWENDY algorithm uses numpy=1.24.3, sklearn=1.4.2, torch=2.2.2

LEQWENDY algorithm uses numpy=1.24.3, sklearn=1.4.2, torch=2.2.2, transformers=4.48.1, peft=0.14.0

please go to https://zenodo.org/records/14927010 to download three large weight files and put them in the 'weights' folder

manuscript for QWENDY method: QWENDY.pdf

_____________________

major code files:

QWENDY_tutorial_OLD.py: a tutorial for using QWENDY, LEQWENDY, TEQWENDY methods 

to apply QWENDY method (old), it also needs the following files: methods.py

to apply TEQWENDY method, it also needs the following files: methods.py, models.py, weights/teqwendy_1st.pth, weights/teqwendy_2nd.pth

to apply LEQWENDY method, it also needs the following files: methods.py, models.py, weights/config.json, weights/pytorch_model.bin, weights/leqwendy_1st.pth, weights/leqwendy_2nd.pth (the last three not uploaded here)

models.py: contains all models

methods.py: contains functions for different methods

_____________________

code files for training other models:

train_teqwendy_first_half.py: train the first half of TEQWENDY method. the trained weights teqwendy_1st.pth is in the folder weights

train_teqwendy_second_half.py: train the second half of TEQWENDY method. the trained weights teqwendy_2nd.pth is in the folder weights

train_leqwendy_first_half.py: train the first half of LEQWENDY method. the trained weights leqwendy_1st.pth is in the folder weights (not uploaded here)

train_leqwendy_second_half.py: train the second half of LEQWENDY method. the trained weights leqwendy_2nd.pth is in the folder weights (not uploaded here)

_____________________

code files for comparing different methods:

test_SINC.py: used to compare different methods on SINC data

test_DREAM4.py: used to compare different methods on DREAM4 data

test_THP1.py: used to compare different methods on THP-1 data

test_hESC.py: used to compare different methods on hESC data

test_mESC.py: used to compare different methods on mESC data

test_hESC_100_qwendy.py: used to test QWENDY on the 100-gene hESC data set

evaluation.py: compare the inferred GRN with the ground truth GRN and calculate AUROC and AUPRC

_____________________

data sets and model weights:

folder SINC: SINC data set. GRN (SINC_A.npy), covariance matrix (SINC_cov.npy), mean level (SINC_mean.npy), from https://figshare.com/articles/software/TRENDY_method_code_files/28236074

folder DREAM4: GRNs and corresponding expression data from https://www.synapse.org/#!Synapse:syn3049712/wiki/74628, has 10 numpy matrices (GRNs) DREAM4_A....npy and 10 numpy matrices (expression data) DREAM4_data....npy

folder THP1: GRN (THP1_A.npy) and corresponding expression data (THP1_data.npy) from https://link.springer.com/article/10.1186/gb-2013-14-10-r118

folder hESC: 18 genes, GRN (hESC_A.npy) and corresponding expression data (hESC_data....npy for six time points) from https://link.springer.com/article/10.1186/s13059-016-1033-x

folder hESC_100: 100 genes, GRN (hESC_A.npy) and corresponding expression data (hESC_data....npy for six time points) from https://link.springer.com/article/10.1186/s13059-016-1033-x

folder mESC: GRN (mESC_new_A.npy) and corresponding expression data (mESC_new_data....npy for five time points) from https://www.nature.com/articles/s41467-018-02866-0

folder training_data: from https://figshare.com/articles/software/TRENDY_method_code_files/28236074, dataset_xx with a number xx means training data; dataset_val means validation data. _A is ground truth GRN; _Kdata is covariance matrix; _Kstar is revised covariance matrix K*; _leqwendyA is the inferred GRN by the first half of LEQWENDY; _teqwendyA is the inferred GRN by the first half of TEQWENDY; _xdata is mean expression level

example_data.npy: example data set, used for QWENDY_tutorial.py

_____________________

folder weights: 

config.json: configuration of Roberta large model

pytorch_model.bin: pre-trained weights of Roberta large model (not uploaded here. please download at https://zenodo.org/records/14927010 and put it in this folder)

leqwendy_1st.pth: weights of the first half of LEQWENDY model (not uploaded here. please download at https://zenodo.org/records/14927010 and put it in this folder)

leqwendy_2nd.pth: weights of the second half of LEQWENDY model (not uploaded here. please download at https://zenodo.org/records/14927010 and put it in this folder)

teqwendy_1st.pth: weights of the first half of TEQWENDY model

teqwendy_2nd.pth: weights of the second half of TEQWENDY model

