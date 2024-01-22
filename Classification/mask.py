import json
import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm

"""
1. 遍历每个图片，每个instance的mask
2. 将polygon区域转化为 mask->boundingrect mask
3. 用boundingrect mask 取截取图片，保存

"""

def random_shift_bbox(bbox, image_shape):
    """
    Randomly shifts a bounding box while keeping its size within the image boundaries.

    Parameters:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        tuple: New bounding box coordinates (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Generate random shifts within the image boundaries
    max_shift_x = image_shape[1] - bbox_width
    max_shift_y = image_shape[0] - bbox_height

    new_x1 = np.random.randint(0, max_shift_x + 1)
    new_y1 = np.random.randint(0, max_shift_y + 1)

    new_x2 = new_x1 + bbox_width
    new_y2 = new_y1 + bbox_height

    return [new_x1, new_y1, new_x2, new_y2]


def create_masked_images(annotation_file, image_dir, mask_dir_anomaly, mask_dir_normal):
    """
    Create binary masks for images based on COCO annotations.
    
    Args:
    annotation_file (str): Path to the COCO annotation file.
    image_dir (str): Directory containing the original images.
    mask_dir (str): Directory where the masked images will be saved.
    """
    # Load annotations
    coco = COCO(annotation_file)
    num = 0
    # Create mask directory if it doesn't exist
    if not os.path.exists(mask_dir_normal):
        os.makedirs(mask_dir_normal)
    if not os.path.exists(mask_dir_anomaly):
        os.makedirs(mask_dir_anomaly)

    # Process each image
    for img_id in tqdm(coco.imgs):
        img_info = coco.imgs[img_id]
        img_path = os.path.join(image_dir, img_info['file_name'])

        # Read image to get its size
        image = cv2.imread(img_path)
        if image is None:
            continue
        height, width = image.shape[:2]

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Draw each annotation on the mask
        for ann in annotations:
            # Create a blank mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            if 'segmentation' in ann:
                bbox = ann['bbox']
                masked_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                cv2.imwrite(mask_dir_anomaly + f'{num}.jpg', masked_img)
                
                new_bbox = random_shift_bbox(bbox, image.shape[:2])
                masked_img = image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :]
                cv2.imwrite(mask_dir_normal + f'{num}.jpg', masked_img)
                num += 1
                


# Example usage
annotation_file = '/opt/data/private/zsf/Railway/part2/after_aug_(0, 3554).json'
image_dir = '/opt/data/private/zsf/Railway/part2/out_harmonization/'
mask_dir_anomaly = '/opt/data/private/zsf/VST_railway/Classification/dataset/anomaly/'
mask_dir_normal = '/opt/data/private/zsf/VST_railway/Classification/dataset/normal/'

create_masked_images(annotation_file, image_dir, mask_dir_anomaly, mask_dir_normal)
