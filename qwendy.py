"""
QWENDY package functions for inferring gene regulatory networks (GRN).
"""

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import GraphicalLassoCV
import warnings

def data_process(data: List[NDArray[np.float64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Process gene expression data to calculate covariance and mean.
    
    Args:
        data: List of 4 numpy arrays, each with shape (n_cells, n_genes)
        n_cells for different arrays can be different
        for more than 4 arrays, only keep the first 4
    Returns:
        K_data: Covariance matrices, shape (4, n_genes, n_genes)
        x_data: Mean expression levels, shape (4, n_genes)
    """
    if len(data) < 4:
        raise ValueError("Data must contain at least 4 time points")
    data = data[:4] # for more than 4 time points, only keep the first 4
    
    K_data = []
    x_data = []
    
    for i in range(4):
        data_i = np.array(data[i], dtype=np.float64)
        x = np.mean(data_i, axis=0)
        try:
            temp = GraphicalLassoCV().fit(data_i)
            K = temp.covariance_
        except:
            K = np.cov(data_i, rowvar=False)
        K_data.append(K)
        x_data.append(x)
    
    K_data = np.array(K_data, dtype=np.float64)
    x_data = np.array(x_data, dtype=np.float64)
    
    return K_data, x_data


def qwendy_kx(K_data: NDArray[np.float64], x_data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    QWENDY method to infer GRN from covariance and mean data.
    
    Args:
        K_data: Covariance matrices, shape (4, n_genes, n_genes)
        x_data: Mean expression levels, shape (4, n_genes)
    
    Returns:
        B: Inferred GRN matrix, shape (n_genes, n_genes)
    """
    K = np.array(K_data, dtype=np.float64)
    x = np.array(x_data, dtype=np.float64)
    
    warning_sign = False # whether any check fails
    
    def make_positive_definite(mat: NDArray[np.float64], epsilon: float = 1e-6) -> NDArray[np.float64]:
        """Make matrix positive definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        num_neg = np.sum(eigenvalues < 1e-8)
        if num_neg > 0:
            small_positives = epsilon * np.arange(1, num_neg + 1)
            eigenvalues[eigenvalues < 1e-8] = small_positives
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Process covariance matrices
    for i in range(4):
        K[i] = (K[i] + K[i].T) / 2  # Make symmetric
        K[i] = make_positive_definite(K[i])  # Make positive definite
        K[i] = (K[i] + K[i].T) / 2  # Make symmetric
    
    # Extract data at 4 time points
    K0, K1, K2, K3 = K[0], K[1], K[2], K[3]
    
    # Check condition (1) in Theorem 1
    for i in range(4):
        if np.linalg.cond(K[i]) > 1e6:
            warning_sign = True
            warnings.warn(f'Warning: K{i} is not invertible (or nearly so)')
    x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
    
    # Cholesky decomposition and matrix operations
    def safe_cholesky(A, base_eps=1e-10, max_tries=5):
        eps = base_eps
        for _ in range(max_tries):
            try:
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                A = A + eps * np.eye(A.shape[0], dtype=A.dtype)
                eps *= 10
        raise np.linalg.LinAlgError("Cholesky failed despite jitter.")
    L0 = safe_cholesky(K0)
    L1 = safe_cholesky(K1)
    L0_inv = np.linalg.inv(L0)
    L1_inv = np.linalg.inv(L1)
    
    # Check condition (2) in Theorem 1
    d1, P1 = np.linalg.eigh(L0_inv @ K1 @ L0_inv.T)
    d1.sort()
    if not np.all(np.diff(d1) > 1e-8):
        warning_sign = True
        warnings.warn('Warning: L0_inv @ K1 @ L0_inv.T has repeated eigenvalues')
    d2, P2 = np.linalg.eigh(L1_inv @ K2 @ L1_inv.T)
    P2_inv = np.linalg.inv(P2)
    d2.sort()
    if not np.all(np.diff(d2) > 1e-8):
        warning_sign = True
        warnings.warn('Warning: L1_inv @ K2 @ L1_inv.T has repeated eigenvalues')
    
    # Calculate transformation matrices
    G = P2_inv @ L1_inv @ K3 @ L1_inv.T @ P2_inv.T
    H = P1.T @ L0_inv @ K2 @ L0_inv.T @ P1
    C = G * H
    
    
    # Find leading eigenvector and determine sign
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    leading_index = np.argmax(eigenvalues)
    leading_eigenvector = eigenvectors[:, leading_index]
    w_vec = np.where(leading_eigenvector >= 0, 1, -1)
    W = np.diag(w_vec)
    
    # Check condition (3) in Theorem 1
    eigenvalues.sort()
    if not np.all(np.diff(eigenvalues) > 1e-8):
        warning_sign = True
        warnings.warn('Warning: C has repeated eigenvalues')
    
    # If not all conditions are satisfied, print a warning
    if warning_sign == True:
        warnings.warn('Warning: At least one condition in Theorem 1 fails. Result might not be reliable')
    
    # Calculate final GRN matrix
    O = P2 @ W @ P1.T
    B = L0_inv.T @ O.T @ L1.T
    
    # Calculate Total Squared Error for both B and -B
    TSE_B = (np.linalg.norm(x1 - x0 @ B) ** 2 + 
              np.linalg.norm(x2 - x1 @ B) ** 2 + 
              np.linalg.norm(x3 - x2 @ B) ** 2 - 
              3 * np.linalg.norm((x1 - x0 @ B + x2 - x1 @ B + x3 - x2 @ B) / 3) ** 2)
    
    TSE_nB = (np.linalg.norm(x1 + x0 @ B) ** 2 + 
               np.linalg.norm(x2 + x1 @ B) ** 2 + 
               np.linalg.norm(x3 + x2 @ B) ** 2 - 
               3 * np.linalg.norm((x1 + x0 @ B + x2 + x1 @ B + x3 + x2 @ B) / 3) ** 2)
    
    return B if TSE_B <= TSE_nB else -B


def qwendy_data(data: List[NDArray[np.float64]],
                print_res: bool = False,
                threshold_upper: float = 1.0,
                threshold_lower: float = -1.0,
                gene_names: Optional[List[str]] = None) -> NDArray[np.float64]:
    """
    QWENDY method to calculate GRN from gene expression data.
    
    Args:
        data: List of 4 numpy arrays, each with shape (n_cells, n_genes)
        n_cells for different arrays can be different
        for more than 4 arrays, only keep the first 4
        print_res: Whether to print inferred regulations
        threshold_upper: Result higher than this threshold means a positive regulation
        threshold_lower: Result lower than this threshold means a negative regulation
        gene_names: List of gene names
    
    Returns:
        B: Inferred GRN matrix, shape (n_genes, n_genes)
    """
    K_data, x_data = data_process(data)
    B = qwendy_kx(K_data, x_data)
    
    if print_res == True:
        """
        if the number of provided gene names is less than actual
        gene number, add gene names
        if the number of provided gene names is more than actual
        gene number, delete extra gene names
        """
        if gene_names is None:
            gene_names = []
        counter = 0
        while len(gene_names) < len(B):
            gene_names.append(f'Gene_{counter}')
            counter += 1
        gene_names = gene_names[:len(B)]
        print('\nGene names: ', gene_names)
        print('Threshold for positive regulation: ', threshold_upper)
        print('Threshold for negative regulation: ', threshold_lower)
        
        results = []
        for i in range(len(B)):
            for j in range(len(B)):
                if i == j:
                    continue 
                results.append([B[i, j], i, j])
        results.sort(reverse=True, key=lambda x:x[0])
        print('\nPositive Regulations:')
        for [reg, i, j] in results:
            if reg > threshold_upper:
                print(gene_names[i] + ' --> ' + gene_names[j])
        results.sort(reverse=False, key=lambda x:x[0])
        print('\nNegative Regulations:')
        for [reg, i, j] in results:
            if reg < threshold_lower:
                print(gene_names[i] + ' --| ' + gene_names[j])
            
    return B


# Package exports
#__all__ = ['data_process', 'qwendy_kx', 'qwendy_data']

#__all__ = ['data_process', 'qwendy_kx', 'qwendy_infer']