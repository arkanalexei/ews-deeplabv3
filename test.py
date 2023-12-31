import torch
import glob
import os

from methods.proposed import proposed_model
from methods import base
from helpers.helpers_test import extract_metadata, transform_image, visualize_results
from helpers.helpers_test_external import default_metadata


# Load the model
model = proposed_model()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Save output
path = 'data/augmented'
paths = glob.glob(path + '/*6_augmented_1.png')
mask_paths = glob.glob(path + '/*6_mask_augmented_1.png')
paths.sort()
mask_paths.sort()
all_paths = zip(paths, mask_paths)

for image_path, mask_path in all_paths:
    # Load image
    # image, metadata = base.BenchmarkMethod.read_image(image_path, rgb=True, metadata=True)
    image = base.BenchmarkMethod.read_image(image_path, rgb=True)
    mask = base.BenchmarkMethod.read_image(mask_path, rgb=False)
    image_transformed = transform_image(image)

    # Extract metadata of training image
    # iso, fnumber, exposure = extract_metadata(metadata)
    iso, fnumber, exposure = default_metadata()

    # Prediction
    model_input = {
        'image': image_transformed.to('cpu'),
        'iso': iso,
        'fnumber': fnumber,
        'exposure': exposure
    }

    with torch.no_grad():
        output = model(model_input)

    filename = os.path.basename(image_path).split('.')[0]
    visualize_results(image, mask, output, filename)
