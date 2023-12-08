import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from helpers.helper_fcts import EncodedToTensor, Normalize, ScaleImage, PadInput


def transform_image(image):
    """
    Transform image as dictated by the referenced paper.

    Keyword arguments:
    image -- The image itself.
    """
    image_pil = Image.fromarray((image * 255).astype('uint8'))
    image_dict = {'image': image_pil}

    transforms = []
    transforms.append(EncodedToTensor())
    transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transforms.append(ScaleImage(2.0))
    transforms.append(PadInput((704, 704)))

    for transform in transforms:
        image_dict = transform(image_dict)

    return image_dict['image'].unsqueeze(0)


def extract_metadata(metadata):
    """
    Extract the ISO, Fnumber, and Exposure from image metadata.

    Keyword arguments:
    metadata -- Information stored inside the image.
    """

    iso = np.float(metadata['iso'])
    iso = np.log2(iso / 100)

    fnumber = metadata['fnumber']
    if '/' in fnumber:
        num, den = fnumber.split('/')
        fnumber = np.float((float(num) / float(den)))
    else:
        fnumber = np.float(fnumber)

    exposure = metadata['exposure']
    if '/' in exposure:
        num, den = exposure.split('/')
        exposure = np.float((float(num) / float(den)))
    else:
        exposure = np.float(exposure)

    iso = torch.tensor([iso])
    fnumber = torch.tensor([fnumber])
    exposure = torch.tensor([exposure])

    return iso, fnumber, exposure

def visualize_results(image, mask, output, filename):
    """
    Visualize segmentation result and compare to original mask

    Keyword arguments:
    image -- Original image
    mask -- Original image
    output -- Segmentation result
    filename -- Filename to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Original Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Original Mask")
    axes[1].axis('off')

    # Segmentation Result
    segmentation_result = torch.argmax(output['mask'].squeeze(0), dim=0)
    axes[2].imshow(segmentation_result.cpu(), cmap='gray')
    axes[2].set_title("Segmentation Result")
    axes[2].axis('off')

    plt.savefig(f'results/{filename}.png')
    plt.close(fig)