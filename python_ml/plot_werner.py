import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the Models
model_filename_p1 = 'svm_poly_degree6_model.pkl' 
model_filename_p2 = 'svm_poly_degree6_model_TARGETED.pkl'

svm_model_p1 = joblib.load(model_filename_p1)
svm_model_p2 = joblib.load(model_filename_p2)
print("Models loaded successfully!")

# Generate 500 Werner states dynamically
p_values = np.linspace(0, 1, 500)
X_werner = np.zeros((500, 9))


# T-matrix singular values = p (features 6, 7, 8) -> set to p
X_werner[:, 6] = p_values
X_werner[:, 7] = p_values
X_werner[:, 8] = p_values


# Calculate f(p) and find exact boundaries for both models
def find_boundary(f_p):
    crossing_idx = np.where(np.diff(np.sign(f_p)))[0][0]
    p1, p2 = p_values[crossing_idx], p_values[crossing_idx + 1]
    f1, f2 = f_p[crossing_idx], f_p[crossing_idx + 1]

    return p1 - f1 * (p2 - p1) / (f2 - f1)


# Phase 1 Predictions
f_p1 = svm_model_p1.decision_function(X_werner)
bound_p1 = find_boundary(f_p1)
print(f"Phase 1 Boundary: p = {bound_p1:.4f}")

# Phase 2 Predictions
f_p2 = svm_model_p2.decision_function(X_werner)
bound_p2 = find_boundary(f_p2)
print(f"Phase 2 Boundary: p = {bound_p2:.4f}")






# Comparative Plot
plt.figure(figsize=(10, 7))

# Phase 1 (Baseline)
plt.plot(p_values, f_p1, label='Phase 1 (Baseline) $f(p)$', 
         color='dodgerblue', linestyle='-.', linewidth=2.5, alpha=0.8)

# Phase 2 (Targeted)
plt.plot(p_values, f_p2, label='Phase 2 (Targeted) $f(p)$', 
         color='navy', linewidth=3)

# Mark the Theoretical Boundary (p = 0.5)
plt.axvline(0.5, color='red', linestyle=':', linewidth=2, label='Theoretical Truth ($p=0.5$)')

# Draw the boundary line at f(p) = 0
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')

# Phase 1 Boundary
plt.plot(bound_p1, 0, marker='o', color='dodgerblue', markersize=8)
plt.annotate(f'P1: $p \\approx {bound_p1:.3f}$', 
             xy=(bound_p1, 0), xytext=(bound_p1 - 0.12, 15),
             color='dodgerblue', fontweight='bold',
             arrowprops=dict(facecolor='dodgerblue', shrink=0.05, width=1, headwidth=5))

# Phase 2 Boundary
plt.plot(bound_p2, 0, marker='o', color='navy', markersize=8)
plt.annotate(f'P2: $p \\approx {bound_p2:.3f}$', 
             xy=(bound_p2, 0), xytext=(bound_p2 + 0.02, 15),
             color='navy', fontweight='bold',
             arrowprops=dict(facecolor='navy', shrink=0.05, width=1, headwidth=5))

# Fill the regions based on p=0.5
plt.axvspan(0, 0.5, facecolor='lightgreen', alpha=0.2, label='True LHS Region')
plt.axvspan(0.5, 1, facecolor='lightcoral', alpha=0.2, label='True Non-LHS Region')


# Formatting
plt.xlabel("Werner State Parameter ($p$)", fontsize=13)
plt.ylabel("SVM Decision Output $f(p)$", fontsize=13)
plt.xlim(0, 1)

min_y = min(min(f_p1), min(f_p2))
max_y = max(max(f_p1), max(f_p2))
plt.ylim(max(min_y, -100), max_y + 30) 

plt.grid(alpha=0.4)
plt.legend(loc='lower left', fontsize=11)
plt.title("Phase 1 vs. Phase 2 Models on Werner States", fontsize=14, fontweight='bold', pad=15)


plt.tight_layout()
plt.show()