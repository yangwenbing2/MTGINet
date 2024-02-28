import torch.nn as nn
import torch.nn.functional as F

from networks.encoders.swin_transformer_enc import SwinTransformer
from networks.ops import SpectralNorm


class SwinTransShortCut_D(SwinTransformer):
    def __init__(self, in_chans=3,
                 patch_size=4,
                 window_size=7,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 out_indices=(0, 1, 2, 3),
                 norm_layer=nn.LayerNorm):
        super(SwinTransShortCut_D, self).__init__(in_channs=in_chans, patch_size=patch_size, window_size=window_size,
                                                  embed_dim=embed_dim, depths=depths, num_heads=num_heads)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.out_indices = out_indices

        # add a norm layer for each output [from Swin-Transformer-Semantic-Segmentation]
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # add shortcut layer [from MG-Matting]
        self.shortcut = nn.ModuleList()
        shortcut_inplanes = [[3, 32], [96, 32], [96, 64], [192, 128], [384, 256], [768, 512]]

        for shortcut in shortcut_inplanes:
            inplane, planes = shortcut
            self.shortcut.append(self._make_shortcut(inplane=inplane, planes=planes, norm_layer=nn.BatchNorm2d))

    def _make_shortcut(self, inplane, planes, norm_layer=nn.BatchNorm2d):
        '''
            came from MGMatting
        '''
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            norm_layer(planes)
        )

    def forward(self, img):
        outs = []
        B_N = img.shape[0]
        outs.append(self.shortcut[0](img))

        # x: [B, L, C]
        x, Wh, Ww = self.patch_embed(img)
        x = self.pos_drop(x)

        x_out = x.transpose(1, 2).view(B_N, -1, Wh, Ww).contiguous()
        outs.append(self.shortcut[1](F.upsample_bilinear(x_out, scale_factor=2.0)))

        for idx, layer in enumerate(self.layers):
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x_out)
                x_out = x_out.transpose(1, 2).view(B_N, -1, H, W).contiguous()
                outs.append(self.shortcut[idx + 2](x_out))

        return outs[-1], {'shortcut': outs[:-1], 'image': img}


if __name__ == '__main__':
    import torch

    print("PyTorch Version:", torch.__version__)

    swin_trans_encoder = SwinTransShortCut_D()
    x = torch.randn(3, 3, 512, 512)
    z = swin_trans_encoder(x)
    print(z[0].shape)
