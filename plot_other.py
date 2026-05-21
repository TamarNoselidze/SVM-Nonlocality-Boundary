import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# Create a 3D grid of t1, t2, t3 values from 0 to 1
t = np.linspace(0, 1, 50)
T1, T2, T3 = np.meshgrid(t, t, t)

# Plug into your exact extracted formula (assuming local vectors are 0)
F = 6.8043 - 1.4690*T1**2 - 2.1996*T2**2 - 3.0298*T3**2 \
    - 3.3489*T1*T2 - 6.4644*T1*T3 - 6.2847*T2*T3

# Use marching cubes to find the surface where F = 0
verts, faces, _, _ = measure.marching_cubes(F, level=0, spacing=(1/49, 1/49, 1/49))

# Plot the 3D surface
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.6, color='cyan', edgecolor='k')
ax.set_xlabel('t_1')
ax.set_ylabel('t_2')
ax.set_zlabel('t_3')
ax.set_title('Machine-Learned LHS Boundary for Bell-Diagonal States')
plt.show()