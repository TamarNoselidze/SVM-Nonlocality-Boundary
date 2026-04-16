import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


from convert_to_T import extract_9d_features, extract_t_matrix_features
# from fidelity import fidelity_kernel


with h5py.File('random_ent_states_array.h5', 'r') as f:
    raw_data = np.array(f['rho'])
    if raw_data.dtype.names is not None and 'r' in raw_data.dtype.names:
        states = raw_data['r'] + 1j * raw_data['i']
    else:
        states = raw_data



with h5py.File('random_ent_labels.h5', 'r') as f:
    raw_columns = np.array(f['labels'])
    if raw_columns.dtype.names is not None and 'l' in raw_columns.dtype.names:
        labels = raw_columns['l'] 
    else:
        labels = raw_columns


# print(labels.shape)
# with h5py.File('is_lhs_from_501_to_1000.h5', 'r') as f:
#     raw_columns_2 = np.array(f['labels'])
#     if raw_columns_2.dtype.names is not None and 'l' in raw_columns_2.dtype.names:
#         labels_2 = raw_columns_2['l'] 
#     else:
#         labels_2 = raw_columns_2


# all_labels = np.concatenate((labels, labels_2)).astype(int)


# filter out ambiguous states
valid_indices = labels != 0
filtered_states = states[valid_indices]
filtered_labels = labels[valid_indices]

print(f"Total states after filtering ambiguous ones (label 0): {len(filtered_labels)}")


# convert nx32 real vectors
def extract_real_features(rho_array):
    n_samples = rho_array.shape[0]
    # Flatten the 4x4 matrices so each state is a 1D array of 16 complex numbers
    flattened = rho_array.reshape(n_samples, -1)
    # Stack the real and imaginary parts side-by-side to get 32 real features
    features = np.hstack((flattened.real, flattened.imag))
    return features



# X = extract_real_features(filtered_states)

X = extract_9d_features(filtered_states)
y = filtered_labels



# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(y_train)} states, Testing on {len(y_test)} states...\n")

# RBF kernel
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
# # svm_model = SVC(decision_function_shape='ovo')

svm_model = SVC(kernel='poly', degree=3)


svm_model.fit(X_train, y_train)


# evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-LHS (-1)", "LHS (1)"]))  





##########################################################################################
# Extracting stuff from the trained model


SVs = svm_model.support_vectors_
weights = svm_model.dual_coef_[0]
bias = svm_model.intercept_[0]
gamma = svm_model._gamma
r = svm_model.coef0

# the constant term  C = b + r^2 * sum(weights)
constant_term = bias + (r**2) * np.sum(weights)

# linear coefficients  L = 2 * gamma * r * sum(w_k * x_k)
linear_coeffs = 2 * gamma * r * np.dot(weights, SVs)

# quadratic coefficients  Q = gamma^2 * sum(w_k * x_k^T * x_k)
Q_matrix = (gamma**2) * np.dot(SVs.T, weights[:, np.newaxis] * SVs)




print("=========================================")
print("THE ANALYTICAL NONLOCALITY FORMULA")
print("f(x) > 0 means LHS, f(x) < 0 means Non-LHS")
print("=========================================\n")

formula_str = f"f(x) = {constant_term:.4f}\n"

# Print linear terms
for i in range(9):
    if abs(linear_coeffs[i]) > 1e-5: # filter out zeros
        sign = "+" if linear_coeffs[i] > 0 else "-"
        formula_str += f"       {sign} {abs(linear_coeffs[i]):.4f} * x_{i+1}\n"

# Print quadratic terms (diagonal: x_1^2, x_2^2)
for i in range(9):
    if abs(Q_matrix[i, i]) > 1e-5:
        sign = "+" if Q_matrix[i, i] > 0 else "-"
        formula_str += f"       {sign} {abs(Q_matrix[i, i]):.4f} * x_{i+1}^2\n"

# Print cross-terms (x_1*x_2, x_3*x_4)
# Because Q is symmetric, we combine the upper and lower triangle (multiply by 2)
for i in range(9):
    for j in range(i + 1, 9):
        cross_coeff = 2 * Q_matrix[i, j]
        if abs(cross_coeff) > 1e-5:
            sign = "+" if cross_coeff > 0 else "-"
            formula_str += f"       {sign} {abs(cross_coeff):.4f} * x_{i+1}*x_{j+1}\n"



print(formula_str)