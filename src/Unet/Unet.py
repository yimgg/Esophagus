import torch
import torch.nn as nn
from monai.networks.nets import UNet

class u_net(UNet):
    pass

if __name__ == '__main__':
    device = 'cuda:0'
    
    x = torch.randn(size=(2, 3, 352, 352)).to(device)
    # test_x = torch.randn(size=(2, 64, 88, 88)).to(device)
    
    model = u_net(spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(4, 8, 16, 32, 64),
                strides=(2, 2, 2, 2)).to(device)
    # module = AttentionBlock(in_dim=64, out_dim=32, kernel_size=3, mlp_ratio=4, shallow=True).to(device)
    
    print(model(x).size())
    # print(module(test_x).size())