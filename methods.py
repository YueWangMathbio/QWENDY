"""
functions of QWENDY, LEQWENDY, TEQWENDY to infer the GRN
"""



from models import Leqwendy_first_half, Leqwendy_second_half, Teqwendy_first_half, Teqwendy_second_half
import torch
import numpy as np
from sklearn.covariance import GraphicalLassoCV
import warnings
warnings.filterwarnings("ignore")

def data_process(data): # from gene express data, calculate covariance and mean
    K_data = []
    x_data = []
    for i in range(4):
        data_i = np.array(data[i])
        x = np.mean(data_i, axis=0)
        temp = GraphicalLassoCV().fit(data_i)
        K = temp.covariance_ 
        K_data.append(K)
        x_data.append(x)
    K_data = np.array(K_data)
    x_data = np.array(x_data)
    return K_data, x_data

def Qwendy_Kx(K_data, x_data): 
    # QWENDY method that calculates GRN from covarianc and mean
    K = np.array(K_data) # covariance
    x = np.array(x_data) # mean level
    def make_positive_definite(mat, epsilon=1e-6): 
        # make each K_i symmetric and positive definite
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        num_neg = np.sum(eigenvalues < 1e-8)
        if num_neg > 0:
            small_positives = epsilon * np.arange(1, num_neg + 1)
            eigenvalues[eigenvalues < 1e-8] = small_positives
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    for i in range(4):
        K[i] = (K[i] + K[i].T) / 2 
        K[i] = make_positive_definite(K[i])
    K0, K1, K2, K3 = K[0], K[1], K[2], K[3]
    x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
    L0 = np.linalg.cholesky(K0)
    L1 = np.linalg.cholesky(K1)
    L0_inv = np.linalg.inv(L0)
    L1_inv = np.linalg.inv(L1)
    d1, P1 = np.linalg.eigh(L0_inv@K1@L0_inv.T)
    d2, P2 = np.linalg.eigh(L1_inv@K2@L1_inv.T)
    P2_inv = np.linalg.inv(P2)
    G = P2_inv @ L1_inv @ K3 @ L1_inv.T @ P2_inv.T
    H = P1.T @ L0_inv @ K2 @ L0_inv.T @ P1
    C = G * H
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    leading_index = np.argmax(eigenvalues)
    leading_eigenvector = eigenvectors[:, leading_index]
    w_vec = np.where(leading_eigenvector >= 0, 1, -1)
    W = np.diag(w_vec)
    O = P2 @ W @ P1.T
    B = L0_inv.T @ O.T @L1.T
    TSE_B = np.linalg.norm(x1-x0@B) ** 2 + np.linalg.norm(x2-x1@B) ** 2
    + np.linalg.norm(x3-x2@B) ** 2 
    - 3 * np.linalg.norm((x1-x0@B+x2-x1@B+x3-x2@B)/3) ** 2
    TSE_nB = np.linalg.norm(x1+x0@B) ** 2 + np.linalg.norm(x2+x1@B) ** 2
    + np.linalg.norm(x3+x2@B) ** 2 
    - 3 * np.linalg.norm((x1+x0@B+x2+x1@B+x3+x2@B)/3) ** 2
    if TSE_B <= TSE_nB:
        return B
    else:
        return -B

def Qwendy_data(data):
    # QWENDY method that calculates GRN from expression data
    K_data, x_data = data_process(data)
    return Qwendy_Kx(K_data, x_data)

def Leqwendy_Kx(K_data, x_data): # 4*n*n and 4*n
    # LEQWENDY method that calculates GRN from covarianc and mean
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = Leqwendy_first_half().to(device)
    state_dict = torch.load('weights/leqwendy_1st.pth', map_location=device)
    model1.load_state_dict(state_dict)
    model1.eval()
    K_data = torch.tensor(K_data, dtype=torch.float32, device=device) # 4*n*n
    K_data = K_data.unsqueeze(0) # 1*4*n*n
    kstar = model1(K_data).cpu().detach().numpy() # 1*4*n*n
    kstar = kstar[0] # 4*n*n
    grn = Qwendy_Kx(kstar, x_data) # n*n
    grn = torch.tensor(grn, dtype=torch.float32, device=device)
    grn = grn.unsqueeze(0)
    grn = grn.unsqueeze(1) # 1*1*n*n
    total_data = torch.concat((K_data, grn), dim=1) # 1*5*n*n
    
    model2 = Leqwendy_second_half().to(device)
    state_dict = torch.load('weights/leqwendy_2nd.pth', map_location=device)
    model2.load_state_dict(state_dict)
    model2.eval()
    teqwendy = model2(total_data).cpu().detach().numpy() # 1*n*n
    teqwendy = teqwendy[0] # n*n
    return teqwendy

def Leqwendy_data(data):
    # LEQWENDY method that calculates GRN from expression data
    K_data, x_data = data_process(data)
    return Leqwendy_Kx(K_data, x_data)

def Teqwendy_Kx(K_data, x_data): # 4*n*n and 4*n
    # TEQWENDY method that calculates GRN from covarianc and mean
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = Teqwendy_first_half().to(device)
    state_dict = torch.load('weights/teqwendy_1st.pth', map_location=device)
    model1.load_state_dict(state_dict)
    model1.eval()
    K_data = torch.tensor(K_data, dtype=torch.float32, device=device) # 4*n*n
    K_data = K_data.unsqueeze(0) # 1*4*n*n
    kstar = model1(K_data).cpu().detach().numpy() # 1*4*n*n
    kstar = kstar[0] # 4*n*n
    grn = Qwendy_Kx(kstar, x_data) # n*n
    grn = torch.tensor(grn, dtype=torch.float32, device=device)
    grn = grn.unsqueeze(0)
    grn = grn.unsqueeze(1) # 1*1*n*n
    total_data = torch.concat((K_data, grn), dim=1) # 1*5*n*n
    
    model2 = Teqwendy_second_half().to(device)
    state_dict = torch.load('weights/teqwendy_2nd.pth', map_location=device)
    model2.load_state_dict(state_dict)
    model2.eval()
    teqwendy = model2(total_data).cpu().detach().numpy() # 1*n*n
    teqwendy = teqwendy[0] # n*n
    return teqwendy

def Teqwendy_data(data):
    # TEQWENDY method that calculates GRN from expression data
    K_data, x_data = data_process(data)
    return Teqwendy_Kx(K_data, x_data)