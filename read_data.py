import h5py
import numpy as np

with h5py.File('entangled_states.h5', 'r') as f:
    raw_data = np.array(f['rho'])
    
    if raw_data.dtype.names is not None and 'r' in raw_data.dtype.names:
        data = raw_data['r'] + 1j * raw_data['i']
    else:
        data = raw_data
        
 
print(f"Total dataset shape: {data.shape}") 

rho_0 = data[0]
print(f"First state shape: {rho_0.shape}")

print(rho_0)
print(f"Trace: {np.trace(rho_0):.5f}") 
print(f"Trace is close to 1: {np.isclose(np.trace(rho_0), 1.0)}")
print(f"Hermitian: {np.allclose(rho_0, rho_0.conj().T)}")
