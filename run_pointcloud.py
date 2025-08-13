import numpy as np
import matplotlib.pyplot as plt

# Load your depth map (replace with your actual loading code)
depth = np.load('depth.npy')  # Example: load a saved depth array

h, w = depth.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Flatten arrays for scatter
ax.scatter(x.flatten(), y.flatten(), depth.flatten(), c=depth.flatten(), cmap='Spectral_r', s=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')
plt.show()