import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


from convert_to_T import extract_9d_features, extract_t_matrix_features



## UNCOMMENT THIS FOR PHASE 1 DATASET
# with h5py.File('random_ent_states_array.h5', 'r') as f:
#     all_states = np.array(f['rho'][:])
# with h5py.File('random_ent_labels.h5', 'r') as f:
#     all_labels = np.array(f['labels'][:])


## PHASE 2 DATASET
with h5py.File("master_lhs_normal.h5", 'r') as f:
    lhs_normal_states = f['rho'][:]
    lhs_normal_labels = f['labels'][:]

with h5py.File("master_non_lhs_normal.h5", 'r') as f:
    nlhs_normal_states = f['rho'][:]
    nlhs_normal_labels = f['labels'][:]


with h5py.File("states_lhs_boundary.h5", 'r') as f:
    lhs_boundary_states = np.array(f['rho'][:])

with h5py.File('labels_lhs_boundary.h5', 'r') as f:
    lhs_boundary_labels = np.array(f['labels'][:])


with h5py.File("master_non_lhs_boundary.h5", 'r') as f:
    nlhs_boundary_states = f['rho'][:]
    nlhs_boundary_labels = f['labels'][:]

all_states = np.concatenate((lhs_normal_states, nlhs_normal_states, lhs_boundary_states, nlhs_boundary_states), axis=0)
all_labels = np.concatenate((lhs_normal_labels, nlhs_normal_labels, lhs_boundary_labels, nlhs_boundary_labels)).astype(int)



print(f"Total states: {all_states.shape[0]}")
print(f"Data type: {all_states.dtype}")


# filter out ambiguous states
valid_indices = all_labels != 0
filtered_states = all_states[valid_indices]
filtered_labels = all_labels[valid_indices]

print(f"Total states after filtering ambiguous ones (label 0): {len(filtered_labels)}")


# convert nx32 real vectors
def extract_real_features(rho_array):
    n_samples = rho_array.shape[0]
    # Flatten the 4x4 matrices so each state is a 1D array of 16 complex numbers
    flattened = rho_array.reshape(n_samples, -1)
    # Stack the real and imaginary parts side-by-side to get 32 real features
    features = np.hstack((flattened.real, flattened.imag))
    return features




# X = extract_real_features(filtered_states)                  # UNCOMMENT THIS FOR 32-D REPRESENTATION
# X = extract_t_matrix_features(filtered_states)              # UNCOMMENT THIS FOR 3-D REPRESENTATION, I.E. ONLY USING T-MATRIX SINGULAR VALUES
X = extract_9d_features(filtered_states)                      # OPTIMAL, 9-D REPRESENTATION OF T-MATRIX SINGULAR VALUES AND LOCAL BLOCH VECTORS
y = filtered_labels



# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {len(y_train)} states, Testing on {len(y_test)} states...\n")

degree=2                #degree of the polynomial kernel
svm_model = SVC(kernel='poly', degree=degree)
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

# linear terms
for i in range(9):
    if abs(linear_coeffs[i]) > 1e-5: # filter out zeros
        sign = "+" if linear_coeffs[i] > 0 else "-"
        formula_str += f"       {sign} {abs(linear_coeffs[i]):.4f} * x_{i+1}\n"

# quadratic terms (diagonal: x_1^2, x_2^2)
for i in range(9):
    if abs(Q_matrix[i, i]) > 1e-5:
        sign = "+" if Q_matrix[i, i] > 0 else "-"
        formula_str += f"       {sign} {abs(Q_matrix[i, i]):.4f} * x_{i+1}^2\n"

# cross-terms (x_1*x_2, x_3*x_4)
for i in range(9):
    for j in range(i + 1, 9):
        cross_coeff = 2 * Q_matrix[i, j]
        if abs(cross_coeff) > 1e-5:
            sign = "+" if cross_coeff > 0 else "-"
            formula_str += f"       {sign} {abs(cross_coeff):.4f} * x_{i+1}*x_{j+1}\n"

print(formula_str)



## Uncomment to save the current model
# model_filename = f'svm_poly_degree{degree}_model_TARGETED.pkl'
# joblib.dump(svm_model, model_filename)
# print(f"\nModel successfully saved to {model_filename}")

