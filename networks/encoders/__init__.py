import logging
from .resnet_enc import ResNet_D, BasicBlock
from .res_shortcut_enc import ResShortCut_D
from.swin_trans_shortcut_enc import SwinTransShortCut_D

__all__ = ['res_shortcut_encoder_29', 'swin_transformer_encoder']


def _res_shortcut_D(block, layers, **kwargs):
    model = ResShortCut_D(block, layers, **kwargs)
    return model


def res_shortcut_encoder_29(**kwargs):
    """Constructs a resnet_encoder_25 model.
    """
    return _res_shortcut_D(BasicBlock, [3, 4, 4, 2], **kwargs)


def _swin_transformer_shortcut(**kwargs):
    model = SwinTransShortCut_D(in_chans=3,
                                patch_size=4,
                                window_size=7,
                                embed_dim=96,
                                depths=(2, 2, 6, 2),
                                num_heads=(3, 6, 12, 24),
                                **kwargs)
    return model


def swin_transformer_encoder(**kwargs):
    """Constructs a swin_transformer_encoder model
    """
    return _swin_transformer_shortcut(**kwargs)


if __name__ == "__main__":
    import torch

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    resnet_encoder = res_shortcut_encoder_29()
    x = torch.randn(4, 3, 512, 512)
    z = resnet_encoder(x)
    print(z[0].shape)
