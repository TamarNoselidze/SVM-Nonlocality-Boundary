import numpy as np

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
paulis = [X, Y, Z]


def extract_9d_features(rho_array): # Singular values of T + a and b vectors
    n_samples = rho_array.shape[0]
    features = np.zeros((n_samples, 9))
    
    for k in range(n_samples):
        rho = rho_array[k]
        
        a_vec = np.zeros(3, dtype=float)
        b_vec = np.zeros(3, dtype=float)
        T = np.zeros((3, 3), dtype=float)
        
        for i in range(3):
            # Alice's vector 
            obs_A = np.kron(paulis[i], I)
            a_vec[i] = np.trace(rho @ obs_A).real
            
            # Bob's vector
            obs_B = np.kron(I, paulis[i])
            b_vec[i] = np.trace(rho @ obs_B).real
            
            # Build the 3x3 T-matrix
            for j in range(3):
                obs_AB = np.kron(paulis[i], paulis[j])
                T[i, j] = np.trace(rho @ obs_AB).real
                
        # Calculate the singular values of T 
        singular_values = np.linalg.svd(T, compute_uv=False)
        
        # Combine everything into a single 9-element array
        features[k] = np.concatenate((a_vec, b_vec, singular_values))
        
    return features



def extract_t_matrix_features(rho_array): # just T matrix singular values

    n_samples = rho_array.shape[0]
    features = np.zeros((n_samples, 3))
    
    for k in range(n_samples):
        rho = rho_array[k]
        T = np.zeros((3, 3), dtype=float)
        
        # 3x3 T-matrix
        for i in range(3):
            for j in range(3):
                observable = np.kron(paulis[i], paulis[j])

                # T_ij = Tr(rho * (sigma_i tensor sigma_j))
                expectation_val = np.trace(rho @ observable)
                T[i, j] = expectation_val.real 
                
        # Calculate the singular values of T
        singular_values = np.linalg.svd(T, compute_uv=False)
        features[k] = singular_values
        
    return features



