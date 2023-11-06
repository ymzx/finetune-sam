import sys
sys.path.append('/data/persist/jwduan/segment-anything-main')  

import torch
import numpy as np
from statistics import mean
from torch.nn.functional import threshold, normalize
from segment_anything.utils.transforms import ResizeLongestSide
from finetune.preprocess import handle_data
from finetune.load_model import model

# 超参设置
lr = 1e-4
wd = 0
device = 'cuda'
num_epochs = 50
model_path = '/data/persist/jwduan/segment-anything-main/finetune/model_files/finetuned.pth'


# 优化器和损失函数
optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn = torch.nn.MSELoss()

# 数据预处理
transformed_data, bbox_coords, ground_truth_masks = handle_data(sam_model=model)
keys = list(bbox_coords.keys())

losses = []
transform = ResizeLongestSide(target_length=1024)
for epoch in range(num_epochs):
  epoch_losses = []
  # Just train on the first 20 examples
  for k in keys[:20]:
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']
    
    # No grad here as we don't want to optimise the encoders
    with torch.no_grad():
      image_embedding = model.image_encoder(input_image)
      
      prompt_box = bbox_coords[k]
      box = transform.apply_boxes(prompt_box, original_image_size)
      box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
      box_torch = box_torch[None, :]
      
      sparse_embeddings, dense_embeddings = model.prompt_encoder(
          points=None,
          boxes=box_torch,
          masks=None,
      )
    low_res_masks, iou_predictions = model.mask_decoder(
      image_embeddings=image_embedding,
      image_pe=model.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=False,
    )

    upscaled_masks = model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
    
    loss = loss_fn(binary_mask, gt_binary_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_losses.append(loss.item())
  losses.append(epoch_losses)
  print(f'EPOCH: {epoch}')
  print(f'Mean loss: {mean(epoch_losses)}')
torch.save(model.state_dict(), model_path)
