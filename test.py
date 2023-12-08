import torch

from methods.proposed import proposed_model
from methods import base
from helpers.helpers_test import extract_metadata, transform_image, visualize_results


img = 'FPWW0180049_RGB1_20170303_133243_6'
image_path = f'data/validation/{img}.png'
mask_path = f'data/validation/{img}_mask.png'

image, metadata = base.BenchmarkMethod.read_image(image_path, rgb=True, metadata=True)
mask = base.BenchmarkMethod.read_image(mask_path, rgb=False)
image_transformed = transform_image(image)

# Load the model
model = proposed_model()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract metadata of training image
iso, fnumber, exposure = extract_metadata(metadata)

# Prediction
model_input = {
    'image': image_transformed.to('cpu'),
    'iso': iso,
    'fnumber': fnumber,
    'exposure': exposure
}

with torch.no_grad():
    output = model(model_input)
    print("output", output)


# Visualize original image and segmentation result
visualize_results(image, mask, output)
