import open3d as o3d

# 读取mesh
mesh = o3d.io.read_triangle_mesh("reconstruction.ply")

# 计算法向量用于更好的渲染效果
mesh.compute_vertex_normals()

# 可视化
o3d.visualization.draw_geometries([mesh])