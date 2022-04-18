import os

import numpy as np
import open3d as o3d
from pathlib import Path

BIN_DATA_DIRECTORY = '2011_09_28/lidar/'
PCD_DATA_DIRECTORY = '2011_09_28/point_clouds/'

def convert_lidar_data_from_bin_to_pcd(file_number_str, remove_bin_file=False):
    bin_file_path = BIN_DATA_DIRECTORY + file_number_str + '.bin'
    pcd_file_path = PCD_DATA_DIRECTORY + file_number_str + '.pcd'

    # Load binary point cloud
    bin_pcd = np.fromfile(bin_file_path, dtype=np.float32)

    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 4))[:, 0:3]

    # Convert to Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    # Save to whatever format you like. Here it saves as pcd
    o3d.io.write_point_cloud(pcd_file_path, o3d_pcd)

    if(remove_bin_file):
        os.remove('bin_file_path')

def convert_all_lidar_data_from_bin_to_pcd(remove_bin_file=False):
    bin_files = os.fsencode(BIN_DATA_DIRECTORY)

    for bin_file in os.listdir(bin_files):
        filename = os.fsdecode(bin_file)
        file_number = Path(filename).stem
        convert_lidar_data_from_bin_to_pcd(file_number, remove_bin_file)

convert_all_lidar_data_from_bin_to_pcd()
