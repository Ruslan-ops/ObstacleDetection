import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# pcd_data_path= "data/2011_09_28/39/point_clouds/"
# file_number_str="0000000274.pcd"


def remove_ground(pcd):
    rest_points, plane_model = get_points_out_of_ground_plane(pcd)
    [a, b, c, d] = plane_model
    #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    plane_norm = np.array([a, b, c])
    np_rest_points = np.asarray(rest_points.points)
    np_rest_points = np_rest_points[np_rest_points @ plane_norm >= -d]
    rest_points.points = o3d.utility.Vector3dVector(np_rest_points)

    #inliers = np.asarray(inliers.points)
    return rest_points, plane_model


def get_points_out_of_ground_plane(point_cloud):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20., max_nn=2), fast_normal_computation=True)
    #pcd.paint_uniform_color([0.6, 0.6, 0.6])
    #o3d.visualization.draw_geometries([pcd]) #Works only outside Jupyter/Colab

    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    return outlier_cloud, plane_model


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

