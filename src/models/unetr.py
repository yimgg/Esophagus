import monai
import numpy as np
import timm
import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.layers import Conv

from src import utils


class UnetrEncoder(nn.Module):
    def __init__(self,
                 image_size=128,
                 patch_size=16,
                 in_channels=4,
                 dropout=0.3,
                 embed_dim=768,
                 num_heads=12,
                 depth=12) -> None:
        super().__init__()

        self.patch_embedding = Conv[Conv.CONV, 3](in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

        img_size = monai.utils.ensure_tuple_rep(image_size, 3)
        pat_size = monai.utils.ensure_tuple_rep(patch_size, 3)
        self.num_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, pat_size)])
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(embed_dim, num_heads, drop_path=dropout, attn_drop=dropout)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 4, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x

    def forward(self, imgs, mask=None):
        x = self.patch_embedding(imgs)
        if mask is not None:
            x = self.mask_model(x, mask)

        x = x.flatten(2).transpose(-1, -2)
        x = x + self.position_embeddings
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # cls_token准备与x进行拼接 B, 1, 1024
        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x)

        # apply Transformer blocks
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        #    cls_token, embedding, hidden_states_out
        # return x[:, :1], x[:, 1:], hidden_states_out
        return x, hidden_states_out


class UnetrDecoder(nn.Module):
    def __init__(self,
                 encoder_output_dim=1024,
                 num_patches=1125,
                 out_channels=4,
                 patch_size=16,
                 decoder_embed_dim=768,
                 decoder_depth=8, decoder_num_heads=16,
                 dropout=0.3
                 ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.decoder_embed = nn.Linear(encoder_output_dim, decoder_embed_dim, bias=True)  # 编码器的最后输出给到解码器的最初输入上

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(decoder_embed_dim, decoder_num_heads, drop=dropout, drop_path=dropout, attn_drop=dropout)
            for i in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred_1 = nn.Linear(decoder_embed_dim, patch_size ** 3, bias=True)  # decoder to patch，中间那个参数是patch的w*h*C（patch的像素）映射回图像
        self.decoder_pred_2 = nn.Linear(decoder_embed_dim, patch_size ** 3, bias=True)  # decoder to patch，中间那个参数是patch的w*h*C（patch的像素）映射回图像
        self.decoder_pred_3 = nn.Linear(decoder_embed_dim, patch_size ** 3, bias=True)  # decoder to patch，中间那个参数是patch的w*h*C（patch的像素）映射回图像
        self.decoder_pred_4 = nn.Linear(decoder_embed_dim, patch_size ** 3, bias=True)  # decoder to patch，中间那个参数是patch的w*h*C（patch的像素）映射回图像

    def forward(self, embedding):
        # embed tokens
        x = self.decoder_embed(embedding)  # Linear,此时传进来的x是latent（保留下来的图像）
        # append mask tokens to sequence, B和L上扩维。
        # mask_tokens.shape[1](mask的L) = ids_restore.shape[1]（for whole img_emb的L） + 1（for cls_token）- x.shape[1]（for latent的L）
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # （latent + mask）no cls token，因为下一行作unsheffle时，ids_restore没有cls_token，所以此时先不考虑cls_token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle，还原成patch的原始顺序
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token，至此x = cls_token+ latent + mask

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_1 = self.decoder_pred_1(x)
        x_2 = self.decoder_pred_2(x)
        x_3 = self.decoder_pred_3(x)
        x_4 = self.decoder_pred_4(x)

        # remove cls token
        x_1 = x_1[:, 1:, :]
        x_2 = x_2[:, 1:, :]
        x_3 = x_3[:, 1:, :]
        x_4 = x_4[:, 1:, :]

        x = torch.cat([x_1, x_2, x_3, x_4], dim=2)

        return x


class SegHeadUnetr(nn.Module):
    def __init__(self, img_size=128,
                 in_channels=4, out_channels=3,
                 feature_size=16, patch_size=16,
                 norm_name="instance",
                 res_block=True, conv_block=True,
                 hidden_size=768,
                 spatial_dims=3):
        super().__init__()
        self.patch_size = monai.utils.ensure_tuple_rep(patch_size, spatial_dims)
        img_size = monai.utils.ensure_tuple_rep(img_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = monai.networks.blocks.dynunet_block.UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, image, encoder_out, hidden_states_out):
        enc1 = self.encoder1(image)
        x2 = hidden_states_out[int(len(hidden_states_out) / 3)][:, 1:]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[int(len(hidden_states_out) / 3 * 2)][:, 1:]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[int(len(hidden_states_out) - 1)][:, 1:]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(encoder_out)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)


class MBotUnetr(nn.Module):
    def __init__(self,
                 image_size=128, patch_size=16, in_channels=4,
                 encoder_num_heads=16, encoder_depth=24,
                 decoder_num_heads=16, decoder_depth=8,
                 embed_dim=1024, dropout=0.3, **kwargs) -> None:
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        # encoder
        self.encoder = UnetrEncoder(image_size=image_size, patch_size=patch_size, in_channels=in_channels,
                                    dropout=dropout, embed_dim=embed_dim, num_heads=encoder_num_heads, depth=encoder_depth)
        self.num_patches = self.encoder.num_patches

        # decoder
        self.decoder = UnetrDecoder(encoder_output_dim=embed_dim, num_patches=self.num_patches, out_channels=in_channels, patch_size=patch_size,
                                    decoder_embed_dim=embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)  # self.apply是nn.module的子函数: 其形参function to be applied to each submodule

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imgs, mask=None):
        embedding, hidden_states_out = self.encoder(imgs, mask)

        pred = self.decoder(embedding)

        return pred, embedding[:, 1:], embedding[:, :1]


class MBotUnetrReconstruct(MBotUnetr):
    '''
    只有一个输出的Mbot
    '''

    def forward(self, imgs, mask=None):
        pred, _, _ = super(MBotUnetrReconstruct, self).forward(imgs, mask)
        img = utils.unpatchify(pred, self.image_size, self.patch_size, self.in_channels)
        return img


class MbotSegUnetr(nn.Module):
    """
    3D分割版 Unetr
    使用Encoder加上一个分割头
    """

    def __init__(self,
                 image_size=128, out_channels=3, patch_size=16, in_channels=4,
                 embed_dim=768, encoder_num_heads=12, encoder_depth=12,
                 dropout=0.3, **kwargs) -> None:
        super().__init__()
        self.image_size = image_size
        # encoder
        self.encoder = UnetrEncoder(image_size=image_size, patch_size=patch_size, in_channels=in_channels,
                                    dropout=dropout, embed_dim=embed_dim,
                                    num_heads=encoder_num_heads, depth=encoder_depth)
        self.seg_hard = SegHeadUnetr(img_size=image_size, in_channels=in_channels, out_channels=out_channels,
                                     feature_size=16, spatial_dims=3, patch_size=patch_size, hidden_size=embed_dim)

    def forward(self, imgs):
        embedding, hidden_states_out = self.encoder(imgs)
        seg_result = self.seg_hard(imgs, embedding[:, 1:], hidden_states_out)
        return seg_result

if __name__ == '__main__':
    x = torch.rand(1,4,128,128,128)
    model = MbotSegUnetr()
    print(model(x).shape)