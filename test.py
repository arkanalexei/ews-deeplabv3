import torch
from methods.proposed import proposed_model
import segmentation_models_pytorch as smp
import torch.optim
import torch.backends.cudnn


model = proposed_model()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

