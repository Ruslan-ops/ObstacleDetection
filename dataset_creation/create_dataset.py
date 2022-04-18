import os
import time
from pathlib import Path
import shutil

import cv2

from projection_to_camera import *
from load_file_functions import *
from image_processing import *
from point_clouds import *

import scipy.ndimage as ndimage

# Set path of image, calibration file, lidar 2011_09_28
image_number = '0000000166'
data_date = '2011_09_28'
object_calib_path = '2011_09_28/calibration/' + image_number + '.txt'
calib_path = '2011_09_28/calibration/' + data_date + '/'
image_path = '2011_09_28/images/' + image_number + '.png'
pcd_path = '2011_09_28/point_clouds/' + image_number + '.pcd'
ground_truth_path = 'data/2011_09_28/39/targets/StixelsGroundTruth.txt'


def create_dataset_simple():
    rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_object_calib_file(object_calib_path)
    calib = read_calib_file(calib_path)

    # Load labels
    #labels = load_label('2011_09_28/000114_label.txt')

    # Load Lidar PC
    pc_velo, pcd = load_pc_velo_scan(pcd_path)
    #pc_velo = remove_ground(pcd)
    pc_velo = pc_velo[:, :3]

    # render_image_with_boxes(rgb, labels, calib)
    #render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    _, velo_pixels, depth, velo_height = render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)

    X, Y, Z = interpolation(velo_pixels[0], velo_pixels[1], depth, (velo_height, rgb.shape[1]))

    Z = ndimage.gaussian_filter(Z, sigma=(3, 1.5), order=0)
    # plt.imshow(Z)
    # plt.show()

    depth_gradient_x = get_depth_gradient(Z, axis=0)
    #plot(X, Y, depth_gradient_x)

    depth_gradient_y = get_depth_gradient(Z, axis=0)

    normed_depth_gradient_y = 255 * depth_gradient_y / np.max(depth_gradient_y) #1. - normalize(depth_gradient_y, -1., 1.)#
    normed_depth_gradient_y[normed_depth_gradient_y > 0] = 0
    normed_depth_gradient_y *= -1

    # plot(X, Y, normed_depth_gradient_y)


    #gravity = np.mean(normed_depth_gradient_y) - 2*np.std(normed_depth_gradient_y)
    #normed_depth_gradient_y = (1000.0/(255.0 - gravity)) * (normed_depth_gradient_y - gravity)
    #normed_depth_gradient_y[normed_depth_gradient_y > 255] = 255

    #np.fromiter((x for x in normed_depth_gradient_y if x < 86), dtype=normed_depth_gradient_y.dtype)
    dl = 1
    dr = 170
    # normed_depth_gradient_y[(normed_depth_gradient_y > dl) & (normed_depth_gradient_y < dr)] = 255
    # normed_depth_gradient_y[normed_depth_gradient_y < dl + 1] = 0
    # normed_depth_gradient_y[normed_depth_gradient_y > dr - 1] = 255
    normed_depth_gradient_y[normed_depth_gradient_y > 0] = 255

    print(np.mean(normed_depth_gradient_y), np.std(normed_depth_gradient_y))
    max_depth = np.max(normed_depth_gradient_y)
    min_depth = np.min(normed_depth_gradient_y)

    # plot(X, Y, normed_depth_gradient_y)

    #normed_depth_gradient_y = cv2.cvtColor(cv2.imread(os.path.join('assets/' + path_name + '.png')), cv2.IMREAD_GRAYSCALE)
    cv2.imshow("some", normed_depth_gradient_y)
    cv2.waitKey(0)

    # plt.hist(normed_depth_gradient_y.ravel(), 256, [0, 256]);
    # plt.show()

    shape = normed_depth_gradient_y.shape
    amount = shape[1]#800#shape[1] // 5;
    stixel_columns = np.array_split(normed_depth_gradient_y, amount, axis=1)
    # cv2.imshow("fj1f", stixel_colomns[100])
    # cv2.waitKey(0)
    stixels = np.zeros(len(stixel_columns))
    rgb = cv2.imread(image_path)
    for i in range(len(stixel_columns)):
        column = stixel_columns[i]
        column_width = column.shape[1]
        column = column[:, 0]
        column = np.reshape(column, -1)
        #column = column[::-1]
        stixels[i] = 0
        for pixel_index in range(len(column) - 1,  75, -1):
            value = column[pixel_index]

            if value > 0:
                stixels[i] = pixel_index
                break


    rgb = cv2.imread(image_path)

    prev_x = 0
    for i in range(len(stixel_columns)):
        column = stixel_columns[i]
        column_width = column.shape[1]
        column_height = column.shape[0]
        color = (0, 255, 0)
        center_x = prev_x + column_width
        prev_x = center_x

        center_y = int(stixels[i]) + img_height - column_height
        cv2.circle(rgb, (center_x, center_y), 4, color=color, thickness=1)

    save_path = '2011_09_28/targets/' + image_number + '.png'
    cv2.imwrite(save_path, rgb)

    cv2.imshow("df", rgb)
    cv2.waitKey(0)

    prev_x, prev_y = 0, 0
    next_x, next_y = 0, 0
    for i in range(1, len(stixel_columns) - 1):
        column = stixel_columns[i]
        column_width = column.shape[1]
        column_height = column.shape[0]
        #posible_stixel = stixels[i]
        center_x = prev_x + column_width
        center_y = int(stixels[i]) + img_height - column_height

        next_x = center_x + stixel_columns[i + 1].shape[1]
        next_y = int(stixels[i + 1]) + img_height - column_height

        critical_distance = 7

        distance_to_prev = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
        if distance_to_prev > critical_distance:
            distance_to_next = np.sqrt((center_x - next_x) ** 2 + (center_y - next_y) ** 2)
            if distance_to_next > critical_distance:
                stixels[i] = 0

        prev_x = center_x
        prev_y = center_y

    rgb = cv2.imread(image_path)
    prev_x = 0
    for i in range(len(stixel_columns)):
        column = stixel_columns[i]
        column_width = column.shape[1]
        column_height = column.shape[0]
        color = (0, 255, 0)
        center_x = prev_x + column_width
        prev_x = center_x

        if stixels[i] > 0:
            center_y = int(stixels[i]) + img_height - column_height
            cv2.circle(rgb, (center_x, center_y), 4, color=color, thickness=1)

    #cv2.imwrite('assets/' + path_name + '_ready.png', rgb)

    cv2.imshow("df", rgb)
    cv2.waitKey(0)
    return stixels, column_height

def create_data_sample(image, calib, lidar_pc, out_data_path):
    img_height, img_width, img_channel = image.shape
    lidar_pc, model = remove_ground(lidar_pc)
    stixels = np.zeros(image.shape[1])

    lidar_pc = np.asarray(lidar_pc.points)
    mask_img = np.zeros((image.shape[0], image.shape[1]))
    mask_img = project_lidar_on_image(mask_img, lidar_pc, calib)
    mask_img = ndimage.gaussian_filter(mask_img, sigma=(1), order=0)

    amount = mask_img.shape[1]
    stixel_columns = np.array_split(mask_img, amount, axis=1)
    for i in range(len(stixel_columns)):
        column = stixel_columns[i]
        column_width = column.shape[1]
        column = column[:, 0]
        column = np.reshape(column, -1)
        # column = column[::-1]
        for pixel_index in range(len(column) - 1, 75, -1):
            value = column[pixel_index]

            if value > 0:
                if pixel_index > stixels[i]:
                    stixels[i] = pixel_index
                    break
    # if not os.path.exists(out_data_path):
    #     os.mknod(out_data_path)

    with open(out_data_path, 'w') as out:
        prev_x = 0
        for i in range(len(stixel_columns)):
            column = stixel_columns[i]
            column_width = column.shape[1]
            column_height = column.shape[0]
            color = (0, 255, 0)
            center_x = prev_x + column_width
            prev_x = center_x

            center_y = int(stixels[i]) + img_height - column_height
            if center_y > 0:
                info = f'{center_x} {center_y}'
                out.write(info + '\n')



def create_dataset_old():
    gtfile = open(ground_truth_path)
    data = gtfile.readlines()
    frame = image_number.lstrip('0')
    points = []
    for line in data:
        line_info = line.split('\t')
        if line_info[0] == '09_26' and line_info[1] == '39' and line_info[2] == frame:
            center_x = int(line_info[3])
            center_y = int(line_info[4])
            points.append((center_x, center_y))

    image = cv2.imread(image_path)
    color = (0, 255, 0)
    for point in points:
        cv2.circle(image, point, 4, color=color, thickness=1)

    cv2.imshow('fjd', image)
    cv2.waitKey(0)


def create_dataset_harder():
    rgb = cv2.imread(image_path)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_object_calib_file(object_calib_path)
    calib = read_calib_file(calib_path)

    # Load labels
    # labels = load_label('2011_09_28/000114_label.txt')

    # Load Lidar PC
    pc_velo, pcd = load_pc_velo_scan(pcd_path)
    #pcd =
    pcd, model = remove_ground(pcd)
    pc_velo = np.asarray(pcd.points)
    #mask = pc_velo[:, 0] > 0
    #pcd.points = o3d.utility.Vector3dVector(pc_velo[mask])
    pc_velo = np.asarray(pcd.points)
    #pc_velo = pc_velo[:, :3]
    clusters = [pc_velo] #get_pc_opject_clusters(pcd)
    pcd_clusters = [pcd] #delete

    # clusters = [cluster for cluster in clusters if is_cluster_fits_in_image(np.stack(cluster), calib, rgb)]
    # pcd_clusters = [o3d.geometry.PointCloud() for i in range(len(clusters))]
    # for (pcd_cluster, cluster) in zip(pcd_clusters, clusters):
    #     pcd_cluster.points = o3d.utility.Vector3dVector(cluster)


    stixels = np.zeros(rgb.shape[1])

    for pc in pcd_clusters:
        pc = np.asarray(pc.points)
        #_, velo_pixels, depth, velo_height = render_lidar_on_image(pc, rgb.copy(), calib, img_width, img_height)
        mask_img = np.zeros((rgb.shape[0], rgb.shape[1]))
        mask_img = project_lidar_on_image(mask_img, pc, calib)
        mask_img = ndimage.gaussian_filter(mask_img, sigma=(1), order=0)

        amount = mask_img.shape[1]
        stixel_columns = np.array_split(mask_img, amount, axis=1)
        for i in range(len(stixel_columns)):
            column = stixel_columns[i]
            column_width = column.shape[1]
            column = column[:, 0]
            column = np.reshape(column, -1)
            #column = column[::-1]
            for pixel_index in range(len(column) - 1, 75, -1):
                value = column[pixel_index]

                if value > 0:
                    if pixel_index > stixels[i]:
                        stixels[i] = pixel_index
                        break

    prev_x = 0
    for i in range(len(stixel_columns)):
        column = stixel_columns[i]
        column_width = column.shape[1]
        column_height = column.shape[0]
        color = (0, 255, 0)
        center_x = prev_x + column_width
        prev_x = center_x

        center_y = int(stixels[i]) + img_height - column_height
        if center_y > 0:
            cv2.circle(rgb, (center_x, center_y), 1, color=color, thickness=1)

    cv2.imshow('fjd', rgb)
    cv2.waitKey(0)

    return stixels, column_height

def create_dataset_for_date(date_path, out_data_path, sample_full_name, remove_source_images=False):
    print(f'--creating for date {date_path}')
    calib_dir_path = os.path.join(date_path, 'calibration')
    calib = read_calib_file(calib_dir_path)
    if calib == None:
        print(f'WARNING: Not found calibration file for date {date_path}. It was skipped')
        return
    date_dir = os.fsencode(date_path)
    for series in os.listdir(date_dir):
        series_name = os.fsdecode(series)
        if series_name == 'calibration':
            continue
        series_path = os.path.join(date_path, series_name)
        create_dataset_for_series(series_path, calib, out_data_path, sample_full_name + series_name + '_', remove_source_images)


def create_dataset_for_series(series_path, calib, out_data_path, sample_full_name, remove_source_images=False):
    print(f'---creating for series {series_path}')
    source_images_path = os.path.join(series_path, 'images')
    source_lidar_path = os.path.join(series_path, 'lidar')

    out_targets_path = os.path.join(out_data_path, 'targets')
    if not os.path.exists(out_targets_path):
        os.makedirs(out_targets_path)

    out_images_path = os.path.join(out_data_path, 'images')
    if not os.path.exists(out_images_path):
        os.makedirs(out_images_path)

    images = os.fsencode(source_images_path)
    for file in os.listdir(images):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            data_number = Path(filename).stem
            lidar_file = os.path.join(source_lidar_path, data_number + '.bin')
            image_path = os.path.join(source_images_path, filename)
            lidar_pc = load_pc_velo_scan(lidar_file)
            if lidar_pc == None:
                print(f"WARNING: No point cloud file related to image {image_path}. It was skipped")
                continue
            image = cv2.imread(image_path)
            full_name = sample_full_name + data_number
            out_target_path = os.path.join(out_targets_path, full_name + '.txt')
            create_data_sample(image, calib, lidar_pc, out_target_path)
            out_image_path = os.path.join(out_images_path, full_name + '.png')

            if remove_source_images:
                Path(image_path).replace(out_image_path)
            else:
                shutil.copyfile(image_path, out_image_path)
            print(f'----created data sample {out_target_path}')
            # print(os.path.join(directory, filename))
            continue
        else:
            print(f"WARNING: Skipped image with name {filename}. Allowed extentions: .png")
            continue

def create_dataset(source_data_path, remove_source_images=False):
    out_data_path = 'dataset'
    kitti_dates = os.fsencode(source_data_path)
    print(f'-dataset creation started from {source_data_path}')
    for date_dir in os.listdir(kitti_dates):
        date_name = os.fsdecode(date_dir)
        date_path = os.path.join(source_data_path, date_name)
        sample_full_name = date_name + '_'
        create_dataset_for_date(date_path, out_data_path, sample_full_name, remove_source_images)
    print('creating annotations')
    create_dataset_annotations(out_data_path)
    print('finished')


def create_dataset_annotations(dataset_path):
    annotations_path = os.path.join(dataset_path, 'annotations.txt')
    images_dir = os.path.join(dataset_path, 'images')
    targets_dir = os.path.join(dataset_path, 'targets')
    with open(annotations_path, 'w') as file:
        images = os.fsencode(images_dir)
        for index, image in enumerate(os.listdir(images)):
            sample_name = Path(os.fsdecode(image)).stem
            image_path = os.path.join(images_dir, sample_name + '.png')
            target_path = os.path.join(targets_dir, sample_name + '.txt')
            if os.path.exists(target_path):
                info = f'{index}\t{image_path}\t{target_path}'
                file.write(info + '\n')
            else:
                print(f'ERROR: No target file in dataset for image {image_path}. It was skipped. Create target file for the image, or delete the image')
                continue




# def prepare_environment(kitti_data_path):
#     if not os.path.exists(kitti_data_path):
#         raise Exception(f'ERROR: Directory with name {kitti_data_path} does not exist')
#     kitti_dir = os.fsencode(kitti_data_path)
#     for date_dir in os.listdir(kitti_dir):
#         for path, subdirs, files in os.walk(date_dir):
#             for name in files:
#                 print(os.path.join(path, name))
#         data_dirs = os.listdir(date_dir)







if __name__ == '__main__':
    create_dataset('source_data', remove_source_images=False)

    #create_dataset_old()
    # t1 = time.time()
    # create_dataset_harder()
    # t2 = time.time()
    # print("Time=", t2 - t1)
















