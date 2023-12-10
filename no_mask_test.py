import torch
import glob
import os

from methods.proposed import proposed_model
from methods import base
from helpers.helpers_test_external import default_metadata, transform_image_external, visualize_results_external


# Load the model
model = proposed_model()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Save output
path = 'data/manual/'
paths = glob.glob(path + '*.jpg')

for image_path in paths:
    # Load image
    image = base.BenchmarkMethod.read_image(image_path, rgb=True)
    image_tensor, image_transformed = transform_image_external(image)

    # Extract metadata of training image
    iso, fnumber, exposure = default_metadata()

    # Prediction
    model_input = {
        'image': image_tensor.to('cpu'),
        'iso': iso,
        'fnumber': fnumber,
        'exposure': exposure
    }

    with torch.no_grad():
        output = model(model_input)

    filename = os.path.basename(image_path).split('.')[0]
    visualize_results_external(image_transformed, output, filename)
