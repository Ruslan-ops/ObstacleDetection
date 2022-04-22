import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# pcd_data_path= "data/2011_09_28/39/point_clouds/"
# file_number_str="0000000274.pcd"

DISTANCE_THRESHOLD = 0.25


def remove_ground(pcd):
    rest_points, plane_model = get_points_out_of_ground_plane(pcd)
    #o3d.visualization.draw_geometries([rest_points]) #Works only outside Jupyter/Colab

    [a, b, c, d] = plane_model
    #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    plane_norm = np.array([a, b, c])
    np_rest_points = np.asarray(rest_points.points)
    np_rest_points = np_rest_points[np_rest_points @ plane_norm >= -d]
    rest_points.points = o3d.utility.Vector3dVector(np_rest_points)

    #inliers = np.asarray(inliers.points)
    return rest_points, plane_model

def remove_high_points(point_cloud, height=1.):
    #o3d.visualization.draw_geometries([point_cloud])
    np_points = np.asarray(point_cloud.points)
    np_points = np_points[np_points[:,2] < height]
    point_cloud.points = o3d.utility.Vector3dVector(np_points)
    #o3d.visualization.draw_geometries([point_cloud])
    return point_cloud


def get_point_cloud_size(point_cloud, axis=0):
    np_pc = np.asarray(point_cloud.points)
    pc_size = (np_pc[:, axis].min(), np_pc[:, axis].max())
    return pc_size

def get_plane_point_cloud(size, plane_model):
    x = np.linspace(size[0], size[1], 300)
    mesh_x, mesh_y = np.meshgrid(x, x)
    (a, b, c, d) = plane_model
    z = (-a * mesh_x + b * mesh_y - d) * (1/c)
    # = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z, -1)
    plane_pc = o3d.geometry.PointCloud()
    plane_pc.points = o3d.utility.Vector3dVector(xyz)
    return plane_pc


def get_points_out_of_ground_plane(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20., max_nn=2), fast_normal_computation=True)
    #pcd.paint_uniform_color([0.6, 0.6, 0.6])
    #o3d.visualization.draw_geometries([point_cloud]) #Works only outside Jupyter/Colab

    plane_model, inliers = point_cloud.segment_plane(distance_threshold=DISTANCE_THRESHOLD, ransac_n=3, num_iterations=2000)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    return outlier_cloud, plane_model

def remove_far_points(point_cloud, max_distance=30):
    np_points = np.asarray(point_cloud.points)
    sqr_sum = np.sum((np_points ** 2), axis=1)
    filter = (sqr_sum < max_distance ** 2) & (sqr_sum > 3. ** 2)
    np_points = np_points[filter]
    rest = o3d.geometry.PointCloud()
    rest.points = o3d.utility.Vector3dVector(np_points)
    #o3d.visualization.draw_geometries([rest])

    return rest

def normalize(vector):
    normalized_v = vector / np.sqrt(np.sum(vector**2))
    return normalized_v

def add_projected_points(plane_model, point_cloud):
    a, b, c, d = plane_model
    plane_norm = np.array([a, b, c])
    plane_unit_norm = normalize(plane_norm)
    np_outlier_cloud = np.asarray(point_cloud.points)
    np_projected_outlier = []
    for vector in np_outlier_cloud:
        vector = vector
        scalar = np.dot(vector, plane_unit_norm)
        projected = vector - scalar * plane_unit_norm
        projected = projected - d * plane_unit_norm
        np_projected_outlier.append(projected)

        extra_ptojectins = _stack_projections(projected, plane_unit_norm, d, 10)
        np_projected_outlier.extend(extra_ptojectins)


    np_projected_outlier = np.r_[np_outlier_cloud, np.array(np_projected_outlier)]
    projected_outlier = o3d.geometry.PointCloud()
    projected_outlier.points = o3d.utility.Vector3dVector(np_projected_outlier)
    return projected_outlier


def _stack_projections(np_projected, plane_unit_norm, d, num_projections=2):
    projections = []
    for i in range(1, num_projections):
        proj_height = i * d / num_projections
        projection = np_projected + proj_height * plane_unit_norm
        projections.append(projection)
    projected2 = np_projected + d * (DISTANCE_THRESHOLD / 2.) * plane_unit_norm
    projections.append(projected2)
    return projections

def get_pc_opject_clusters(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=1., min_points=30))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])
    clustered_pcd = sorted(zip(labels, np.asarray(pcd.points)), key=lambda t: t[0])
    labels = np.asarray(clustered_pcd)[:, 0]
    np_points = np.asarray(clustered_pcd)[:, 1]
    split_indexes = np.unique(labels, return_index=True)[1][1:]
    clusters = np.split(np_points, split_indexes)
    return clusters

