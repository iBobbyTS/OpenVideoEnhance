import cv2
import numpy
import torch
from models import BasicVSRNet
import fix_model

torch.set_grad_enabled(False)
model = BasicVSRNet(
    mid_channels=64,
    num_blocks=30,
    spynet_pretrained='/Users/ibobby/Dataset/model_weights/BasicVSR/spynet.pth'
)
# state_dict = torch.load('/Users/ibobby/Dataset/model_weights/BasicVSR/v-bi-fixed.pth')
state_dict = fix_model.new
model.load_state_dict(state_dict)
model.eval()
# model.to(torch.float16)
img = torch.from_numpy(cv2.imread('/Users/ibobby/Pictures/lr.png').transpose(2, 0, 1).astype(numpy.float32)/255.0).unsqueeze(0)

img = torch.stack([img, img, img], 1)
# img = img.unsqueeze(0)
with torch.no_grad():
    img = (model(img).squeeze()[0].numpy()*255.0).round().astype(numpy.uint8).transpose(1, 2, 0)
cv2.imwrite('/Users/ibobby/PycharmProjects/OpenVideoEnhance/bicubic.png', img)

print('success')
