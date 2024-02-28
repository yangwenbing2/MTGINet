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


if __name__ == "__main__":
    device = torch.device("cuda")
    inputs = torch.rand(2, 3, 512, 512).to(device)

    # swin_transformer_encoder or res_shortcut_encoder_29
    model = get_generator(encoder="res_shortcut_encoder_29", decoder="res_shortcut_decoder_22").to(device)
    pred_trimap, pred_matte, pred_os2, pred_os4, pred_os8 = model(inputs, inference=False)

    print(pred_matte.shape, pred_os2.shape, pred_os4.shape, pred_os8.shape)
    print("pred_trimap:", pred_trimap.shape)

    # ======================= FLOPs and params =======================
    from torchstat import stat
    # from thop import profile
    #
    # # stat(model, input_size=(3, 512, 512))
    #
    # inputs = torch.rand(1, 3, 512, 512).to(device)
    # FLOPs, params = profile(model, (inputs,))
    # GFLOPs, Mparams = FLOPs / (1e9), params / (1e6)
    # print("Gflops:{:.4f}, Mparams:{:.4f}".format(GFLOPs, Mparams))

    # ======================= FPS =======================
    # import time
    # device = torch.device("cuda")
    # model = get_generator(encoder="res_shortcut_encoder_29", decoder="res_shortcut_decoder_22").to(device)
    # model.eval()
    # inputs = torch.randn((1, 3, 512, 512)).to(device)
    # _ = model(inputs)
    #
    # count = 500
    # total_time = 0
    # for i in range(count):
    #     print("\r Processing:[{}/{}]".format(i + 1, count), end="", flush=True)
    #     inputs = torch.randn((1, 3, 512, 512)).to(device)
    #
    #     with torch.no_grad():
    #         start = time.time()
    #         out = model(inputs)
    #         end = time.time()
    #         cost_time = end - start
    #
    #     total_time += cost_time
    #
    # print("\nAverage inference time:{:.6f}, fps:{:.1f}".format(total_time / count, count / total_time))
