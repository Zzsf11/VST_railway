from torchvision import datasets, transforms, models
import cv2
import torch

# x = cv2.imread('/opt/data/private/zsf/VST_railway/RGB_VST/preds/DUTS/RGB_VST/1_cv2diff.png')
model = models.detection.fasterrcnn_resnet50_fpn()
# model.eval()

x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
a = torch.tensor([[100,100,200,200],[100,100,250,200]], dtype=torch.float)
b = torch.tensor([1,2],dtype=torch.int64)
targets = [{'boxes': a, 
       'labels':  b},
      {'boxes': a, 
       'labels':  b}]

predictions = model(x, targets)