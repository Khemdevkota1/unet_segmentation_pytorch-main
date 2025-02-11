from torch.hub import load
from torch.nn import Sigmoid





def unet_model():
    model=load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.add_module("sigmoid",Sigmoid())
    return model
