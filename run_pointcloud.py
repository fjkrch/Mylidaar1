import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def read_ply_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Find header end
    i = 0
    while not lines[i].startswith('end_header'):
        i += 1
    data = np.loadtxt(lines[i+1:])
    return data

# Load point cloud data (XYZ only)
points = read_ply_xyz("depth_points.ply")
x, y, z = points[:,0], points[:,1], points[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=z, cmap='Spectral_r', s=0.5)

ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_zlabel('Depth (AI estimate)')
plt.colorbar(sc, ax=ax, label='Depth')
plt.title('AI Estimated Depth Point Cloud')
plt.show()

# Load the point cloud from file
pcd = o3d.io.read_point_cloud("depth_points.ply")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])