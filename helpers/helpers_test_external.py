import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from helpers.helper_fcts import EncodedToTensor, Normalize, ScaleImage, PadInput


# def crop_image(image_pil, target_size=(350, 350)):
#     """
#     Crop the image to the target size.

#     Keyword arguments:
#     image_pil -- The PIL image to crop.
#     target_size -- The target size for cropping (width, height).
#     """
#     # Calculate dimensions for cropping
#     width, height = image_pil.size
#     left = (width - target_size[0])/2
#     top = (height - target_size[1])/2
#     right = (width + target_size[0])/2
#     bottom = (height + target_size[1])/2

#     # Crop the center of the image
#     image_cropped = image_pil.crop((left, top, right, bottom))
#     return image_cropped

def resize_image(image_pil, target_size=(350, 350)):
    return image_pil.resize(target_size, Image.ANTIALIAS)

def transform_image_external(image):
    """
    Transform image as dictated by the referenced paper.

    Keyword arguments:
    image -- The image itself.
    """
    image_pil = Image.fromarray((image * 255).astype('uint8'))
    image_pil = resize_image(image_pil)
    image_dict = {'image': image_pil}

    transforms = []
    transforms.append(EncodedToTensor())
    transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transforms.append(ScaleImage(2.0))
    transforms.append(PadInput((704, 704)))

    for transform in transforms:
        image_dict = transform(image_dict)

    return image_dict['image'].unsqueeze(0), image_pil


def default_metadata():
    """
    Return default metadata values.
    """
    iso = np.float(100)
    iso = np.log2(iso / 100)
    fnumber = np.float(8)
    exposure = np.float(1/90)

    iso = torch.tensor([iso])
    fnumber = torch.tensor([fnumber])
    exposure = torch.tensor([exposure])

    return iso, fnumber, exposure


def visualize_results_external(image, output, filename):
    """
    Visualize segmentation result

    Keyword arguments:
    image -- Original image
    output -- Segmentation result
    filename -- Filename to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Segmentation Result
    segmentation_result = torch.argmax(output['mask'].squeeze(0), dim=0)
    axes[1].imshow(segmentation_result.cpu(), cmap='gray')
    axes[1].set_title("Segmentation Result")
    axes[1].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0)
    fig.tight_layout(pad=1)

    plt.savefig(f'results/manual/{filename}.png', bbox_inches='tight')
    plt.close(fig)