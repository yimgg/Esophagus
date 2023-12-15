import torch
import time
import yaml
from torch import nn
from easydict import EasyDict
from src.SlimUNETR.SlimUNETR import SlimUNETR


def test_weight(model, x):
    # torch.cuda.synchronize()
    start_time = time.time()
    _ = model(x)
    # torch.cuda.synchronize()
    end_time = time.time()
    # torch.cuda.synchronize()
    need_time = end_time - start_time
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / 10000000, 2)))
    print('flop : {} G'.format(round(flops / 10000000000, 2)))
    print('throughout: {} img/min'.format(throughout * 60))


if __name__ == '__main__':
    # 读取配置
    device = 'cpu'
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    x = torch.rand(1, 3, 352, 352).to(device)
    model = SlimUNETR(**config.slim_unetr).to(device)

    # print(model(x).shape)
    for i in range(0, 10):
        _ = model(x)
    flops, param, throughout = test_weight(model, x)
    Unitconversion(flops, param, throughout)
