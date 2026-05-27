import h5py
import numpy as np

# Lists to hold the data from each batch
all_states = []
all_labels = []

num_batches = 4

print(f"Merging {num_batches} batches...")

for i in range(1, num_batches + 1):
    state_file = f"states_non_lhs_normal_{i}.h5"
    label_file = f"labels_non_lhs_normal_{i}.h5" 
    
    with h5py.File(state_file, 'r') as f_state:
        states = f_state['rho'][:]
        print(states.shape)
        print(states[0].shape)

        all_states.append(states)
        
    # Read the labels
    with h5py.File(label_file, 'r') as f_label:
        labels = f_label['labels'][:]
        all_labels.append(labels)

# Concatenate everything along the first axis 
merged_states = np.concatenate(all_states, axis=0)
merged_labels = np.concatenate(all_labels, axis=0)

print(f"Success! Final merged states shape: {merged_states.shape}")
print(f"Final merged labels shape: {merged_labels.shape}")


with h5py.File("master_non_lhs_normal.h5", 'w') as f_out:
    f_out.create_dataset("rho", data=merged_states)
    f_out.create_dataset("labels", data=merged_labels)
    
print("Saved to 'master_non_lhs_normal.h5'.")