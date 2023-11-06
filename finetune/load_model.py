import sys
sys.path.append('/data/persist/jwduan/segment-anything-main')  

from segment_anything import sam_model_registry

model_type = 'vit_h'
checkpoint = '/data/persist/jwduan/segment-anything-main/checkpoint/sam_vit_h_4b8939.pth'
device = 'cuda'

model = sam_model_registry[model_type](checkpoint=checkpoint)
model.to(device)
model.train()