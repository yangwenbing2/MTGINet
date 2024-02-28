import os
import cv2
import numpy as np

import torch

import utils
from utils import CONFIG
import networks
from utils import compute_sad_loss, compute_connectivity_error, compute_gradient_loss, \
    compute_mse_loss, compute_mad_loss, calculate_sad_mse_mad_whole_img


class Tester(object):
    def __init__(self, test_dataloader, logger):

        self.test_dataloader = test_dataloader
        self.logger = logger

        self.model_config = CONFIG.model
        self.test_config = CONFIG.test
        self.log_config = CONFIG.log
        self.data_config = CONFIG.data

        self.build_model()
        self.resume_step = None

        utils.print_network(self.G, CONFIG.version)

        if self.test_config.checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.test_config.checkpoint))
            self.restore_model(self.test_config.checkpoint)

    def build_model(self):
        self.G = networks.build_model(CONFIG.model.model_arch, pretrained=True)
        if not self.test_config.cpu:
            self.G.cuda()

    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
            :param resume_checkpoint: File name of checkpoint
            :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path)
        self.G.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    def test(self):
        self.G = self.G.eval()
        mse_loss = 0
        sad_loss = 0
        mad_loss = 0
        conn_loss = 0
        grad_loss = 0

        test_num = 0

        count = len(self.test_dataloader)
        with torch.no_grad():
            for idx, image_dict in enumerate(self.test_dataloader):
                image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                alpha_shape, name = image_dict['alpha_shape'], image_dict['image_name']

                if not self.test_config.cpu:
                    image = image.cuda()
                    alpha = alpha.cuda()
                    trimap = trimap.cuda()

                _, _, alpha_pred, _, _, _ = self.G(image)

                if CONFIG.model.trimap:
                    alpha_pred[trimap == 2] = 1
                    alpha_pred[trimap == 0] = 0

                    trimap[trimap == 2] = 255
                    trimap[trimap == 1] = 128

                for cnt in range(image.shape[0]):
                    h, w = alpha_shape
                    test_alpha = alpha[cnt, 0, ...].data.cpu().numpy() * 255
                    test_alpha = test_alpha.astype(np.uint8)
                    test_pred = alpha_pred[cnt, 0, ...].data.cpu().numpy() * 255
                    test_pred = test_pred.astype(np.uint8)

                    test_pred = cv2.resize(test_pred[:, :, None], (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
                    test_alpha = cv2.resize(test_alpha[:, :, None], (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
                    if CONFIG.model.trimap:
                        test_trimap = trimap[cnt, 0, ...].data.cpu().numpy()
                        test_trimap = test_trimap[:h, :w]
                    else:
                        test_trimap = None

                    if self.test_config.alpha_path is not None:
                        cv2.imwrite(os.path.join(self.test_config.alpha_path, os.path.splitext(name[cnt])[0] + ".png"),
                                    test_pred)

                    # mse_diff = compute_mse_loss(test_pred, test_alpha, test_trimap)
                    # mad_diff = compute_mad_loss(test_pred, test_alpha, test_trimap)
                    # sad_diff = compute_sad_loss(test_pred, test_alpha, test_trimap)[0]

                    sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(test_pred / 255., test_alpha / 255.)
                    sad_loss += sad_diff
                    mse_loss += mse_diff
                    mad_loss += mad_diff
                    print("[{}/{}]  sad_loss:{}  mse_loss:{}   mad_loss:{}  Name:{}".format(idx + 1, count, sad_diff, mse_diff, mad_diff, name))

                    if not self.test_config.fast_eval:
                        conn_loss += compute_connectivity_error(test_pred, test_alpha, test_trimap, 0.1)
                        grad_loss += compute_gradient_loss(test_pred, test_alpha, test_trimap)

                    test_num += 1

        self.logger.info("TEST NUM: \t\t {}".format(test_num))
        self.logger.info("SAD: \t\t {}".format(sad_loss / test_num))
        self.logger.info("MSE: \t\t {}".format(mse_loss / test_num))
        self.logger.info("MAD: \t\t {}".format(mad_loss / test_num))

        if not self.test_config.fast_eval:
            self.logger.info("GRAD: \t\t {}".format(grad_loss / test_num))
            self.logger.info("CONN: \t\t {}".format(conn_loss / test_num))
