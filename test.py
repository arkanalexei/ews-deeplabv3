import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from methods.proposed import proposed_model
from helpers.helper_fcts import EncodedToTensor, Normalize, ScaleImage, PadInput


img = 'FPWW0180619_RGB1_20170420_165439_6'

image_path = f'data/validation/{img}.png'
mask_path = f'data/validation/{img}_mask.png'
image = Image.open(image_path)
image_dict = {'image': image}

# Apply the same transforms as during training
transforms = []
transforms.append(EncodedToTensor())
transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
transforms.append(ScaleImage(2.0))
transforms.append(PadInput((704, 704)))

for transform in transforms:
    image_dict = transform(image_dict)

image_transformed = image_dict['image'].unsqueeze(0)

# Load the model
model = proposed_model()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Default values for metadata
iso = torch.tensor([np.log2(100 / 100)])  # log2(iso/100)
fnumber = torch.tensor([8.0])
exposure = torch.tensor([1/125.0])

# Prediction
model_input = {
    'image': image_transformed.to('cpu'),
    'iso': iso,
    'fnumber': fnumber,
    'exposure': exposure
}

with torch.no_grad():
    output = model(model_input)

# Visualize original image and segmentation result
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Original Image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Original Mask
axes[1].imshow(Image.open(mask_path))
axes[1].set_title("Original Mask")
axes[1].axis('off')

# Segmentation Result
segmentation_result = torch.argmax(output['mask'].squeeze(0), dim=0)
axes[2].imshow(segmentation_result.cpu(), cmap='gray')
axes[2].set_title("Segmentation Result")
axes[2].axis('off')

plt.show()