import albumentations as A
import random
from StixelsDataset import *

def visualize_dataset(dataset_path, stixels100=False):
    transform = A.Compose(
        [
            A.Resize(height=400, width=800),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    dataset = StixelsDataset(os.path.join(dataset_path, 'annotations.txt'), dataset_path)
    annotations = dataset.annotations
    for i, (img_name, tar_name) in enumerate(annotations):
        if i % 50 != 0:
            continue
        image_path = os.path.join(dataset.images_path, img_name)
        target_path = os.path.join(dataset.targets_path, tar_name)
        image = cv2.imread(image_path)

        points = dataset._read_target_file(target_path)
        transformed = transform(image=image, keypoints=points)
        #image = transformed['image']
        #points = transformed['keypoints']
        # vis_keypoints(transformed['image'], transformed['keypoints'])
        # image = transform(image)
        # points = target_transform(points)
        draw_stixels(image, points, img_name, stixels100)





def draw_stixels(image, points, img_name, stixels100):
    if stixels100:
        stixel_columns_amount = 100
        targets = np.zeros((stixel_columns_amount), dtype=np.float32)
        for x, y in points:
            index = int((x * stixel_columns_amount - 0.00001) / image.shape[1])
            if y > targets[index]:
                targets[index] = y
        # targets = np.clip(targets, 0.51, 49.49)
        points = []
        for ind, stix in enumerate(targets):
            xcor = int(ind * image.shape[1] / stixel_columns_amount)
            ycor = stix
            points.append((xcor, ycor))
        points = np.array(points)

    for x, y in points:
        if y >= 0:
            x = int(x)
            y = int(y)
            cv2.circle(image, (x, y), 5, color=(0, 255, 0), thickness=-1)
    cv2.imshow(img_name, image)
    cv2.waitKey(0)


visualize_dataset('dataset/', True)

