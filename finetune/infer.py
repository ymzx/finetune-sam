import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/persist/jwduan/segment-anything-main') 

from segment_anything import sam_model_registry, SamPredictor
from finetune.preprocess import handle_data
from segment_anything.build_sam import _build_sam

data_dir = '/data/persist/jwduan/segment-anything-main/finetune/datasets/stamps/'
device = 'cuda'
model_type = 'vit_h'

checkpoint = '/data/persist/jwduan/segment-anything-main/checkpoint/sam_vit_h_4b8939.pth'
sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_orig.to(device)

checkpoint = '/data/persist/jwduan/segment-anything-main/finetune/model_files/finetuned.pth'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)

# 数据预处理
transformed_data, bbox_coords, ground_truth_masks = handle_data(sam_model=sam_model)
keys = list(bbox_coords.keys())


predictor_tuned = SamPredictor(sam_model)
predictor_original = SamPredictor(sam_model_orig)

k = keys[10]
image = cv2.imread(data_dir+f'scans/scans/{k}.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor_tuned.set_image(image)
predictor_original.set_image(image)

input_bbox = np.array(bbox_coords[k])

masks_tuned, _, _ = predictor_tuned.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

masks_orig, _, _ = predictor_original.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

_, axs = plt.subplots(1, 2, figsize=(25, 25))

axs[0].imshow(image)
show_mask(masks_tuned, axs[0])
show_box(input_bbox, axs[0])
axs[0].set_title('Mask with Tuned Model', fontsize=26)
axs[0].axis('off')


axs[1].imshow(image)
show_mask(masks_orig, axs[1])
show_box(input_bbox, axs[1])
axs[1].set_title('Mask with Untuned Model', fontsize=26)
axs[1].axis('off')

plt.savefig('/data/persist/jwduan/segment-anything-main/finetune/seg_result.png')