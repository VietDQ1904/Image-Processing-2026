import polyscope as ps
import numpy as np
import os
from collections import defaultdict

base_dir = os.path.dirname(__file__)

points_path = os.path.join(base_dir, "../local/raw_npz/BALBc_1_points.npy")
edges_path = os.path.join(base_dir, "../local/raw_npz/BALBc_1_edges.npy")
radius_path = os.path.join(base_dir, "../local/raw_npz/BALBc_1_radius.npy")

# load dữ liệu
points = np.load(points_path)
edges = np.load(edges_path)
radis = np.load(radius_path)
radis = radis.reshape(-1)

print("points:", points.shape)
print("edges:", edges.shape)
print("radi: ", radis.shape)


# khởi tạo polyscope
ps.init()

ps.set_ground_plane_mode("none")

# đăng ký graph
network = ps.register_curve_network(
    "vascular_graph_BALBc_1",
    points,
    edges
)

network.set_radius(0.0005)

network.add_scalar_quantity(
    "radius",
    radis,
    defined_on="edges",
    cmap="turbo",
    enabled=True
)

ps.show()