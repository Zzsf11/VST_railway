import os
from PIL import Image
import numpy as np

# Define paths
data_root = '/opt/data/private/zsf/VST/RGB_VST/Data/Mydataset_harmonization_cv2diff/my-Mask'
out_root = '/opt/data/private/zsf/VST/RGB_VST/Data/Mydataset_harmonization_cv2diff/my-Contour'

# Get all image files in the data_root directory
im_files = [f for f in os.listdir(data_root) if f.endswith('.jpg')]

# Process each image file
for im_file in im_files:
    id = im_file[:-4]  # Remove the '.png' extension to get the ID

    # Read the image
    gt_path = os.path.join(data_root, im_file)
    gt = Image.open(gt_path).convert('L')
    gt = np.array(gt) > 128
    gt = gt.astype(np.float64)
    
    
    # Calculate gradient and edge
    gy, gx = np.gradient(gt)
    temp_edge = np.sqrt(gy**2 + gx**2)
    temp_edge[temp_edge != 0] = 1
    bound = (temp_edge * 255).astype(np.uint8)

    # Save the edge image
    save_path = os.path.join(out_root, f'{id}_edge.png')
    Image.fromarray(bound).save(save_path)
