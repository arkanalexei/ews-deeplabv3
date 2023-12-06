import torch
from methods.proposed import DeepLabV3PlusInjNet
import segmentation_models_pytorch as smp
import torch.optim
import torch.backends.cudnn


def proposed_model():
    network = smp.DeepLabV3Plus('resnet50', classes=2)

    name = 'dlb50_inj35ft15'
    inputs = ['image', 'fnumber', 'exposure', 'iso']
    outputs = ['mask']
    inj_layers = [None, None, None, None, None, ['fnumber', 'exposure', 'iso']]
    inj_type = 'concat'

    for mod_name, module in network.encoder.named_children():
        if mod_name in ['layer2', 'layer3']:
            for param in module.parameters():
                param.requires_grad = False

    model = DeepLabV3PlusInjNet(name, inputs, outputs, network, inj_layers, inj_type)

    return model


model = proposed_model()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

