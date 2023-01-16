import torch
torch.manual_seed(3)
import torch.utils.data as data
import numpy as np
from dataloader_mtl import MTLDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import models

#Loading MTL model
mtl_model = models.MTLNet(c_in=3, c_out_seg=1)
mtl_model.load_state_dict(torch.load('mtl_model.pt', map_location=torch.device('cpu')))

#Test dataloader
transforms = transforms.Compose([transforms.ToTensor()])
testset = MTLDataset(type='test', transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
test_iter = iter(testloader)

image, bbox_label, bin_label, mask_label = next(test_iter)
pred_bin, pred_mask, pred_bbox = mtl_model(image.float())

#Removing batch dimension and class dimension
mask_label = mask_label.squeeze(0)
mask_label = mask_label.squeeze(0)
pred_mask = pred_mask.squeeze(0).detach().numpy()
pred_mask = pred_mask.squeeze(0)

fig, ax = plt.subplots(1,2)
ax[0].imshow(mask_label)
ax[1].imshow(pred_mask)
ax[0].axis('off')
ax[1].axis('off')
ax[0].set_title('Target Mask')
ax[1].set_title('Predicted Mask')
plt.savefig('mask.png')