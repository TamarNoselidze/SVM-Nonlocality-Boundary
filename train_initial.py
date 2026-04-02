import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

with h5py.File('entangled_states.h5', 'r') as f:
    # first 1500 states 
    raw_data = np.array(f['rho'][:1500])
    if raw_data.dtype.names is not None and 'r' in raw_data.dtype.names:
        states = raw_data['r'] + 1j * raw_data['i']
    else:
        states = raw_data

# Labels 1-500
with h5py.File('is_lhs_from_1_to_500.h5', 'r') as f:
    raw_columns_1 = np.array(f['labels'])
    if raw_columns_1.dtype.names is not None and 'l' in raw_columns_1.dtype.names:
        labels_1 = raw_columns_1['l'] 
    else:
        labels_1 = raw_columns_1

# Labels 501-1000
with h5py.File('is_lhs_from_501_to_1000.h5', 'r') as f:
    raw_columns_2 = np.array(f['labels'])
    if raw_columns_2.dtype.names is not None and 'l' in raw_columns_2.dtype.names:
        labels_2 = raw_columns_2['l'] 
    else:
        labels_2 = raw_columns_2


# Labels 1001-1500
with h5py.File('is_lhs_from_1001_to_1500.h5', 'r') as f:
    raw_columns_3 = np.array(f['labels'])
    if raw_columns_3.dtype.names is not None and 'l' in raw_columns_3.dtype.names:
        labels_3 = raw_columns_3['l'] 
    else:
        labels_3 = raw_columns_3


# Combine 1500 labels
all_labels = np.concatenate((labels_1, labels_2, labels_3)).astype(int)

# filter out ambiguous states
# valid_indices = all_labels != 0
# filtered_states = states[valid_indices]
# filtered_labels = all_labels[valid_indices]

# print(f"Total states after filtering ambiguous ones (label 0): {len(filtered_labels)}")


# Convert Nx4x4 complex matrices into Nx32 real vectors for the SVM
def extract_real_features(rho_array):
    n_samples = rho_array.shape[0]
    # Flatten the 4x4 matrices so each state is a 1D array of 16 complex numbers
    flattened = rho_array.reshape(n_samples, -1)
    # Stack the real and imaginary parts side-by-side to get 32 real features
    features = np.hstack((flattened.real, flattened.imag))
    return features

# X = extract_real_features(filtered_states)
# y = filtered_labels

X = extract_real_features(all_labels)
y = all_labels


# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(y_train)} states, Testing on {len(y_test)} states...\n")

# RBF kernel
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
svm_model = SVC(decision_function_shape='ovo')

svm_model.fit(X_train, y_train)

# evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-LHS (-1)", "LHS (1)"]))  