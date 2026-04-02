import h5py
import numpy as np

with h5py.File('entangled_states.h5', 'r') as f:
    raw_data = np.array(f['rho'])
    
    if raw_data.dtype.names is not None and 'r' in raw_data.dtype.names:
        data = raw_data['r'] + 1j * raw_data['i']
    else:
        data = raw_data
        

with h5py.File('is_lhs_from_1_to_500.h5', 'r') as f:
    raw_columns = np.array(f['labels'])
    
    if raw_columns.dtype.names is not None and 'l' in raw_columns.dtype.names:
        labels = raw_columns['l'] + 1j * raw_columns['i']
    else:
        labels = raw_columns


print(f"Total dataset shape: {data.shape}") 
print(f"Columns shape: {labels.shape}")

rho_0 = data[0]
print(f"First state shape: {rho_0.shape}")

print(rho_0)


# data_ = data[:500]
# lab = labels[:500]

# lhs_count = 0
# non_lhs_count = 0
# ambiguous_count = 0

# for i in range(500):
#     rho = data[i]
#     label = lab[i]
#     if label == 1:
#         lhs = "lhs"
#         lhs_count+=1
#     elif label == -1:
#         lhs = "not lhs"
#         non_lhs_count+=1
#     else: # 0
#         lhs = "ambiguous"
#         ambiguous_count+=1

#     # print(rho)
#     # print(f"Trace: {np.trace(rho):.5f}") 
#     # print(f"Trace is close to 1: {np.isclose(np.trace(rho), 1.0)}")
#     # print(f"Hermitian: {np.allclose(rho, rho.conj().T)}")
#     # print(f"LHS status: {lhs}")

# print(f'Number of LHS states: {lhs_count}')
# print(f'Number of non LHS states: {non_lhs_count}')
# print(f'Number of ambiguous states: {ambiguous_count}')
