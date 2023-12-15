import torch
import torch.nn as nn
from src.SlimUNETR.Encoder import Encoder
from src.SlimUNETR.Decoder import Decoder

class SlimUNETR(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, embed_dim=96,embedding_dim=64, channels=(24, 48, 60),
                        blocks=(1, 2, 3, 2), heads=(1, 2, 4, 4), r=(4, 2, 2, 1), distillation=False,
                        dropout=0.3):
        super(SlimUNETR, self).__init__()
        self.Encoder = Encoder(in_channels=in_channels, embed_dim=embed_dim,
                                                   embedding_dim=embedding_dim,
                                                   channels=channels,
                                                   blocks=blocks, heads=heads, r=r, distillation=distillation,
                                                   dropout=dropout)
        self.Decoder = Decoder(out_channels=out_channels, embed_dim=embed_dim, channels=channels,
                                      blocks=blocks, heads=heads, r=r, distillation=distillation, dropout=dropout)

    def forward(self, x):
        embeding, hidden_states_out, (B, C, H, W) = self.Encoder(x)
        x = self.Decoder(embeding, hidden_states_out, (B, C, H, W))
        return x


def test_weight(model, x):
    import time
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
    print('params : {} K'.format(round(params*1024 / 10000000, 2)))
    print('flop : {} M'.format(round(flops*1024 / 10000000000, 2)))
    print('throughout: {} FPS'.format(throughout))

if __name__ == '__main__':
    x = torch.randn(size=(2, 3, 352, 352))
    model = SlimUNETR(in_channels=3, out_channels=1, embed_dim=96,embedding_dim=121, channels=(24, 48, 60),
                        blocks=(1, 2, 3, 2), heads=(1, 2, 4, 4), r=(4, 2, 2, 1), distillation=False,
                        dropout=0.3)
    print(model(x).shape)
    for i in range(0,10):
        _ = model(x)
    flops, params, throughout = test_weight(model,x)
    Unitconversion(flops, params, throughout)

