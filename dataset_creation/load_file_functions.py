import os

import cv2
import mayavi.mlab as mlab
import numpy as np
import  open3d as o3d

def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_pc_velo_scan(bin_file_path):
    bin_pcd = np.fromfile(bin_file_path, dtype=np.float32)

    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 4))[:, 0:3]

    # Convert to Open3D point cloud
    o3d_point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    return o3d_point_cloud


def load_pc_velo_scanz(velo_filename):
    if velo_filename.endswith(".pcd") and os.path.exists(velo_filename):
        pcd = o3d.io.read_point_cloud(velo_filename)
        return np.asarray(pcd.points), pcd
    return None, None



def read_object_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def read_calib_file(calib_directiory_path):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    calib_velo_to_cam = os.path.join(calib_directiory_path, 'calib_velo_to_cam.txt')
    if not os.path.exists(calib_velo_to_cam):
        print(f'ERROR: No calibration file {calib_velo_to_cam}')
        return None
    calib_cam_to_cam = os.path.join(calib_directiory_path, 'calib_cam_to_cam.txt')
    if not os.path.exists(calib_cam_to_cam):
        print(f'ERROR: No calibration file {calib_cam_to_cam}')
        return None
    with open(calib_velo_to_cam, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    with open(calib_cam_to_cam, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

