import sys
sys.path.append('/data/persist/jwduan/segment-anything-main')  

data_dir = '/data/persist/jwduan/segment-anything-main/finetune/datasets/stamps/'

from collections import defaultdict
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything.utils.transforms import ResizeLongestSide


transformed_data = defaultdict(dict)

device = 'cuda'

# Exclude scans with zero or multiple bboxes (of the first 100)
stamps_to_exclude = {
    'stampDS-00008',
    'stampDS-00010',
    'stampDS-00015',
    'stampDS-00021',
    'stampDS-00027',
    'stampDS-00031',
    'stampDS-00039',
    'stampDS-00041',
    'stampDS-00049',
    'stampDS-00053',
    'stampDS-00059',
    'stampDS-00069',
    'stampDS-00073',
    'stampDS-00080',
    'stampDS-00090',
    'stampDS-00098',
    'stampDS-00100'
}.union({
    'stampDS-00012',
    'stampDS-00013',
    'stampDS-00014',
}) # Exclude 3 scans that aren't the type of scan we want to be fine tuning for

bbox_coords = {}
ground_truth_masks = {}

def handle_data(sam_model=None):
    for f in sorted(Path(data_dir+'ground-truth-maps/ground-truth-maps/').iterdir())[:100]:
        k = f.stem[:-3]
        if k not in stamps_to_exclude:
            im = cv2.imread(f.as_posix())
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            if len(contours) > 1:
                x,y,w,h = cv2.boundingRect(contours[0])
                height, width, _ = im.shape
                bbox_coords[k] = np.array([x, y, x + w, y + h])

    for k in bbox_coords.keys():
        image = cv2.imread(data_dir+f'scans/scans/{k}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        transform = ResizeLongestSide(target_length=1024)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
        input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[k]['image'] = input_image
        transformed_data[k]['input_size'] = input_size
        transformed_data[k]['original_image_size'] = original_image_size

    
    for k in bbox_coords.keys():
        gt_grayscale = cv2.imread(data_dir+f'ground-truth-pixel/ground-truth-pixel/{k}-px.png', cv2.IMREAD_GRAYSCALE)
        ground_truth_masks[k] = (gt_grayscale == 0)

    return transformed_data, bbox_coords, ground_truth_masks