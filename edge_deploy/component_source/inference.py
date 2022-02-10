import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import torchvision.transforms.functional as TF
from PIL import ImageDraw
import numpy as np
# import matplotlib.pyplot as plt


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


# load image
pred_img = Image.open('FudanPed00001.jpeg')
img = TF.to_tensor(pred_img)
print(img.shape)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2
trained_model = get_model_instance_segmentation(num_classes)

# load model
with open('../model.pth', 'rb') as f:
    trained_model.load_state_dict(torch.load(f))
    trained_model.to(device)
    trained_model.eval()

# do inference
with torch.no_grad():
    prediction = trained_model([img.to(device)])

# visualization
im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
w = max(int(np.shape(im)[0]*0.007), 2)

draw = ImageDraw.Draw(im)
for b in prediction[0]['boxes']:
    draw.rectangle(b.cpu().numpy(), outline=(0, 255, 0), width=w)

im.save("result.png")

# plt.figure(figsize=(20, 15))
# plt.subplots_adjust(wspace=0.0, hspace=0.0)
# col_num = 3
# row_num = len(prediction[0]['masks'])//col_num + 1

for i, m in enumerate(prediction[0]['masks']):
    im = Image.fromarray(m.cpu()[0].mul(127).byte().numpy(), 'L')
    draw = ImageDraw.Draw(im)
    # plt.subplot(row_num, col_num, i+1)
    # plt.subplots_adjust(hspace=0.0)
    # plt.imshow(im, cmap='Blues')
    im.save("mask-" + str(i) + ".png")
    
print("inference pedestrian image is done")
