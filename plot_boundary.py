import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the model
model_filename = 'svm_poly_degree3_model.pkl'
svm_model = joblib.load(model_filename)
print(f"Model '{model_filename}' loaded successfully!")

# 2. Generate 500 Werner states dynamically
# We create an array of 500 states, each with 9 features initialized to 0.
p_values = np.linspace(0, 1, 500)
X_werner = np.zeros((500, 9))

# Werner state physical properties:
# a_vec = 0 (features 0, 1, 2) -> left as 0
# b_vec = 0 (features 3, 4, 5) -> left as 0
# T-matrix singular values = p (features 6, 7, 8) -> set to p
X_werner[:, 6] = p_values
X_werner[:, 7] = p_values
X_werner[:, 8] = p_values

# 3. Calculate f(p) using the SVM's built-in math
# This applies all weights, biases, and degrees automatically!
f_p = svm_model.decision_function(X_werner)

# 4. Dynamically find the exact SVM Boundary (where f(p) crosses 0)
# We find the two points where the sign flips, and interpolate for high precision
crossing_idx = np.where(np.diff(np.sign(f_p)))[0][0]
p1, p2 = p_values[crossing_idx], p_values[crossing_idx + 1]
f1, f2 = f_p[crossing_idx], f_p[crossing_idx + 1]

# Linear interpolation to find the exact root (f(p) = 0)
svm_boundary = p1 - f1 * (p2 - p1) / (f2 - f1)
print(f"Calculated SVM Boundary for Werner States: p = {svm_boundary:.4f}")

# 5. Create the plot
plt.figure(figsize=(8, 6))

# Plot the SVM decision function
plt.plot(p_values, f_p, label='SVM Decision Function: $f(p)$', color='blue', linewidth=2.5)

# Fill the LHS and Non-LHS regions dynamically based on the max/min of the function
plt.axhspan(0, max(f_p) + 1, facecolor='lightgreen', alpha=0.3, label='LHS Region (Unsteerable)')
plt.axhspan(min(f_p) - 1, 0, facecolor='lightcoral', alpha=0.3, label='Non-LHS Region (Steerable)')

# Draw the boundary line at f(p) = 0
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')

# Mark the Machine-Learned Boundary dynamically
plt.plot(svm_boundary, 0, 'ko', markersize=8)
plt.annotate(f'SVM Boundary\n$p \\approx {svm_boundary:.3f}$', 
             xy=(svm_boundary, 0), xytext=(svm_boundary + 0.05, max(f_p)*0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

# Mark the Theoretical Boundary (p = 0.5)
plt.axvline(0.5, color='red', linestyle=':', linewidth=2, label='Theoretical Bound ($p=0.5$)')

# Formatting
plt.title(f"SVM Prediction for Werner States ({svm_model.kernel.capitalize()} Kernel)", fontsize=14, fontweight='bold')
plt.xlabel("Werner State Parameter ($p$)", fontsize=12)
plt.ylabel("SVM Decision Output $f(p)$", fontsize=12)
plt.xlim(0, 1)

# Dynamically set y-limits based on the function's output range
plt.ylim(min(f_p) - 0.5, max(f_p) + 0.5)
plt.grid(alpha=0.4)
plt.legend(loc='lower left')

# Show the plot
plt.tight_layout()
plt.show()