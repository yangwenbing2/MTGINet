import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import encoders, decoders, ops


class Generator(nn.Module):
    def __init__(self, encoder, decoder):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        self.aspp = ops.ASPP(in_channel=512, out_channel=512)

        self.conv_semantic = ops.Conv2dIBNormRelu(512, 3, kernel_size=3, stride=1,
                                                  padding=1, with_ibn=False, with_relu=False)

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, inference=False):
        embedding, mid_fea = self.encoder(image)
        embedding = self.aspp(embedding)

        pred_trimap = None
        if not inference:
            semantic_fea = F.interpolate(embedding, scale_factor=2, mode='bilinear', align_corners=False)
            pred_trimap = self.conv_semantic(semantic_fea)

        pred_alpha, pred_os2, pred_os4, pred_os8 = self.decoder(embedding, mid_fea)

        return pred_trimap, pred_alpha, pred_os2, pred_os4, pred_os8


def get_generator(encoder, decoder):
    generator = Generator(encoder=encoder, decoder=decoder)
    return generator

