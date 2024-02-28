import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.backends.cudnn as cudnn
from torch.nn import SyncBatchNorm
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel

import utils
from utils import CONFIG
import networks


class Trainer(object):
    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 logger,
                 tb_logger):

        # Save GPU memory.
        cudnn.benchmark = False

        # set logger
        self.logger = logger
        self.tb_logger = tb_logger

        # set config
        self.model_config = CONFIG.model
        self.train_config = CONFIG.train
        self.log_config = CONFIG.log

        # set loss
        self.loss_dict = {'trimap': None,
                          'detail': None,
                          'alpha': None,
                          'comp': None}
        self.test_loss_dict = {'mse': None,
                               'sad': None}

        self.gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                          [4., 16., 24., 16., 4.],
                                          [6., 24., 36., 24., 6.],
                                          [4., 16., 24., 16., 4.],
                                          [1., 4., 6., 4., 1.]]).cuda()
        self.gauss_filter /= 256.
        self.gauss_filter = self.gauss_filter.repeat(1, 1, 1, 1)

        # set dataloader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # set optimizer
        self.G_optimizer = None
        self.G_scheduler = None

        # set model
        self.G = None
        self.build_model()
        self.resume_step = None
        self.best_sad_loss = 1e+8
        self.best_mse_loss = 1e+8

        utils.print_network(self.G, CONFIG.version)
        if self.train_config.resume_checkpoint:
            self.logger.info('Resume checkpoint: {}'.format(self.train_config.resume_checkpoint))
            self.restore_model(self.train_config.resume_checkpoint)

        if self.model_config.imagenet_pretrain and self.train_config.resume_checkpoint is None:
            self.logger.info('Load Imagenet Pretrained: {}'.format(self.model_config.imagenet_pretrain_path))

            utils.load_imagenet_pretrain(self.G, self.model_config.imagenet_pretrain_path)

            # YWB: use swin_transformer as backbone
            # utils.load_SwinTransformer_pretrain(self.G, self.model_config.imagenet_pretrain_path)

    def build_model(self):
        self.G = networks.get_generator(CONFIG.model.arch.encoder, CONFIG.model.arch.decoder)
        self.G.cuda()

        if CONFIG.dist:
            self.logger.info("Using pytorch synced BN")
            self.G = SyncBatchNorm.convert_sync_batchnorm(self.G)

        self.G_optimizer = torch.optim.Adam(self.G.parameters(),
                                            lr=self.train_config.G_lr,
                                            betas=(self.train_config.beta1, self.train_config.beta2))

        if CONFIG.dist:
            # SyncBatchNorm only supports DistributedDataParallel with single GPU per process
            self.G = DistributedDataParallel(self.G, device_ids=[CONFIG.local_rank], output_device=CONFIG.local_rank)
        else:
            self.G = nn.DataParallel(self.G)

        self.build_lr_scheduler()

    def build_lr_scheduler(self):
        """Build cosine learning rate scheduler."""
        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                          T_max=self.train_config.total_step -
                                                                self.train_config.warmup_step)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.G_optimizer.zero_grad()

    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path, map_location=lambda storage, loc: storage.cuda(CONFIG.gpu))

        self.resume_step = checkpoint['iter']
        self.logger.info('Loading the trained models from step {}...'.format(self.resume_step))
        self.G.load_state_dict(checkpoint['state_dict'], strict=True)

        if not self.train_config.reset_lr:
            if 'opt_state_dict' in checkpoint.keys():
                try:
                    self.G_optimizer.load_state_dict(checkpoint['opt_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
            else:
                self.logger.info('No Optimizer State Loaded!!')

            if 'lr_state_dict' in checkpoint.keys():
                try:
                    self.G_scheduler.load_state_dict(checkpoint['lr_state_dict'])
                except ValueError as ve:
                    self.logger.error("{}".format(ve))
        else:
            self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
                                                              T_max=self.train_config.total_step - self.resume_step - 1)

        if 'loss' in checkpoint.keys():
            self.best_sad_loss = 1e+8
            self.best_mse_loss = 1e+8

    def train(self):
        data_iter = iter(self.train_dataloader)

        if self.train_config.resume_checkpoint:
            start = self.resume_step + 1
        else:
            start = 0

        moving_max_grad = 0
        moving_grad_moment = 0.999
        max_grad = 0

        # load pretrain model
        from utils import util
        checkpoint_path = "/media/ilab/Innocent/ilab/experiments/MyNet/CFGNet/checkpoints/b2_P3M-10K/best_mse_model.pth"
        ckpt = torch.load(checkpoint_path)
        # self.G.load_state_dict(util.remove_prefix_state_dict(ckpt['state_dict']), strict=True)
        self.G.load_state_dict(ckpt['state_dict'], strict=True)

        for step in range(start, self.train_config.total_step + 1):
            try:
                image_dict = next(data_iter)
            except:
                data_iter = iter(self.train_dataloader)
                image_dict = next(data_iter)

            image, alpha, trimap, fg, bg = image_dict['image'], image_dict['alpha'], image_dict['trimap'], \
                                           image_dict['fg'], image_dict['bg']
            image, alpha, trimap, fb, bg = image.cuda(), alpha.cuda(), trimap.cuda(), fg.cuda(), bg.cuda()

            # train() of DistributedDataParallel has no return
            self.G.train()
            log_info = ""
            loss = 0

            """===== Update Learning Rate ====="""
            if step < self.train_config.warmup_step and self.train_config.resume_checkpoint is None:
                cur_G_lr = utils.warmup_lr(self.train_config.G_lr, step + 1, self.train_config.warmup_step)
                utils.update_lr(cur_G_lr, self.G_optimizer)
            else:
                self.G_scheduler.step()
                cur_G_lr = self.G_scheduler.get_lr()[0]

            """===== Forward G ====="""
            pred_trimap, pred_alpha, pred_os2, pred_os4, pred_os8 = self.G(image)

            weight = utils.get_unknown_tensor(trimap)

            """===== Calculate Loss ====="""
            if self.train_config.semantic_weight > 0:
                self.loss_dict['trimap'] = self.semantic_loss(pred_trimap=pred_trimap,
                                                              trimap=trimap) * self.train_config.semantic_weight

            if self.train_config.detail_weight > 0:
                pred_details = [pred_os2, pred_os4, pred_os8]
                self.loss_dict['detail'] = self.detail_loss(pred_details=pred_details, target=alpha,
                                                            gauss_filter=self.gauss_filter, weight=weight) \
                                           * self.train_config.detail_weight

            if self.train_config.alpha_weight > 0:
                # self.loss_dict['alpha'] = self.alpha_loss(pred_alpha=pred_alpha, target=alpha,
                #                                           gauss_filter=self.gauss_filter, weight=weight) \
                #                           * self.train_config.alpha_weight
                self.loss_dict['alpha'] = self.lap_loss(logit=pred_alpha, target=alpha,
                                                          gauss_filter=self.gauss_filter, weight=weight) \
                                          * self.train_config.alpha_weight

            if self.train_config.comp_weight > 0:
                self.loss_dict['comp'] = self.composition_loss(logit=pred_alpha, image=image,
                                                               fg=fg, bg=bg) * self.train_config.comp_weight

            for loss_key in self.loss_dict.keys():
                if self.loss_dict[loss_key] is not None and loss_key in ['trimap', 'detail', 'alpha', 'comp']:
                    loss += self.loss_dict[loss_key]

            """===== Back Propagate ====="""
            self.reset_grad()

            loss.backward()

            """===== Clip Large Gradient ====="""
            if self.train_config.clip_grad:
                if moving_max_grad == 0:
                    moving_max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 1e+6)
                    max_grad = moving_max_grad
                else:
                    max_grad = nn_utils.clip_grad_norm_(self.G.parameters(), 2 * moving_max_grad)
                    moving_max_grad = moving_max_grad * moving_grad_moment + max_grad * (
                            1 - moving_grad_moment)

            """===== Update Parameters ====="""
            self.G_optimizer.step()

            """===== Write Log and Tensorboard ====="""
            # stdout log
            if step % self.log_config.logging_step == 0:
                # reduce losses from GPUs
                if CONFIG.dist:
                    self.loss_dict = utils.reduce_tensor_dict(self.loss_dict, mode='mean')
                    loss = utils.reduce_tensor(loss)

                # create logging information
                for loss_key in self.loss_dict.keys():
                    if self.loss_dict[loss_key] is not None:
                        log_info += loss_key.upper() + ": {:.7f}, ".format(self.loss_dict[loss_key])

                self.logger.debug("Image tensor shape: {}. Trimap tensor shape: {}".format(image.shape, trimap.shape))
                log_info = "[{}/{}], ".format(step, self.train_config.total_step) + log_info
                log_info += "lr: {:6f}".format(cur_G_lr)
                self.logger.info(log_info)

                # tensorboard
                if step % self.log_config.tensorboard_step == 0 or step == start:  # and step > start:
                    self.tb_logger.scalar_summary('Loss', loss, step)

                    # detailed losses
                    for loss_key in self.loss_dict.keys():
                        if self.loss_dict[loss_key] is not None:
                            self.tb_logger.scalar_summary('Loss_' + loss_key.upper(),
                                                          self.loss_dict[loss_key], step)

                    self.tb_logger.scalar_summary('LearnRate', cur_G_lr, step)

                    if self.train_config.clip_grad:
                        self.tb_logger.scalar_summary('Moving_Max_Grad', moving_max_grad, step)
                        self.tb_logger.scalar_summary('Max_Grad', max_grad, step)

                # write images to tensorboard
                if step % self.log_config.tensorboard_image_step == 0 or step == start:
                    image_set = {'image': (utils.normalize_image(image[-1, ...]).data.cpu().numpy()
                                           * 255).astype(np.uint8),
                                 'trimap': (trimap[-1, ...].data.cpu().numpy() * 127).astype(np.uint8),
                                 'alpha': (alpha[-1, ...].data.cpu().numpy() * 255).astype(np.uint8),
                                 'alpha_pred': (pred_alpha[-1, ...].data.cpu().numpy() * 255).astype(np.uint8)}

                    self.tb_logger.image_summary(image_set, step)

            """===== TEST ====="""
            if (step % self.train_config.val_step) == 0 or step == self.train_config.total_step:  # and step > start:
                test_loss = 0

                self.test_loss_dict['mse'] = 0
                self.test_loss_dict['sad'] = 0
                for loss_key in self.loss_dict.keys():
                    if loss_key in self.test_loss_dict and self.loss_dict[loss_key] is not None:
                        self.test_loss_dict[loss_key] = 0

                self.G.eval()
                with torch.no_grad():
                    for image_dict in self.test_dataloader:
                        image, alpha, trimap = image_dict['image'], image_dict['alpha'], image_dict['trimap']
                        alpha_shape = image_dict['alpha_shape']
                        image, alpha, trimap = image.cuda(), alpha.cuda(), trimap.cuda()

                        # inference
                        _, pred_alpha, _, _, _ = self.G(image)

                        h, w = alpha_shape
                        pred_alpha = pred_alpha[..., :h, :w]
                        trimap = trimap[..., :h, :w]

                        # weight = utils.get_unknown_tensor(trimap)
                        weight = None

                        # value of MSE/SAD here is different from test.py and matlab version
                        self.test_loss_dict['mse'] += self.mse(pred_alpha, alpha, weight)
                        self.test_loss_dict['sad'] += self.sad(pred_alpha, alpha, weight)

                # reduce losses from GPUs
                if CONFIG.dist:
                    self.test_loss_dict = utils.reduce_tensor_dict(self.test_loss_dict, mode='mean')

                """===== Write Log and Tensorboard ====="""
                # stdout log
                log_info = ""
                for loss_key in self.test_loss_dict.keys():
                    if self.test_loss_dict[loss_key] is not None:
                        self.test_loss_dict[loss_key] /= len(self.test_dataloader)
                        test_loss += self.test_loss_dict[loss_key]
                        # logging
                        log_info += loss_key.upper() + ": {:.4f} ".format(self.test_loss_dict[loss_key])
                        self.tb_logger.scalar_summary('Loss_' + loss_key.upper(),
                                                      self.test_loss_dict[loss_key], step, phase='test')

                self.logger.info("TEST: LOSS: {:.4f} ".format(test_loss) + log_info)
                self.tb_logger.scalar_summary('Loss', test_loss, step, phase='test')

                image_set = {'image': (utils.normalize_image(image[-1, ...]).data.cpu().numpy() * 255).astype(np.uint8),
                             'trimap': (trimap[-1, ...].data.cpu().numpy() * 127).astype(np.uint8),
                             'alpha': (alpha[-1, ...].data.cpu().numpy() * 255).astype(np.uint8),
                             'alpha_pred': (pred_alpha[-1, ...].data.cpu().numpy() * 255).astype(np.uint8)}

                self.tb_logger.image_summary(image_set, step, phase='test')

                """===== Save Model ====="""
                if (step % self.log_config.checkpoint_step == 0 or step == self.train_config.total_step) \
                        and CONFIG.local_rank == 0 and (step > start):

                    self.logger.info('Saving the trained models from step {}...'.format(iter))
                    self.save_model("latest_model", step, loss)

                    if self.test_loss_dict['mse'] < self.best_mse_loss:
                        self.best_mse_loss = self.test_loss_dict['mse']
                        self.save_model("best_mse_model", step, loss)

                    if self.test_loss_dict['sad'] < self.best_sad_loss:
                        self.best_sad_loss = self.test_loss_dict['sad']
                        self.save_model("best_sad_model", step, loss)

    @staticmethod
    def mse(logit, target, weight=None):
        if weight is not None:
            return F.mse_loss(logit * weight, target * weight)
        else:
            return F.mse_loss(logit, target)

    @staticmethod
    def sad(logit, target, weight=None):
        if weight is not None:
            return F.l1_loss(logit * weight, target * weight, reduction='sum') / 1000
        else:
            return F.l1_loss(logit, target, reduction='sum') / 1000

    def save_model(self, checkpoint_name, iter, loss):
        """Restore the trained generator and discriminator."""
        torch.save({
            'iter': iter,
            'loss': loss,
            'state_dict': self.G.state_dict(),
            'opt_state_dict': self.G_optimizer.state_dict(),
            'lr_state_dict': self.G_scheduler.state_dict()
        }, os.path.join(self.log_config.checkpoint_path, '{}.pth'.format(checkpoint_name)))

    @staticmethod
    def composition_loss(logit, image, fg, bg):
        """
        Alpha composition loss
        """
        weight = torch.ones(logit.shape).cuda()

        comp = logit * fg + (1 - logit) * bg
        diff = torch.sqrt((comp - image) ** 2 + 1e-12)
        comp_loss = diff.sum() / weight.sum()

        return comp_loss

    @staticmethod
    def lap_loss(logit, target, gauss_filter, weight=None):
        '''
        Based on FBA Matting implementation:
        https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
        '''

        def conv_gauss(img, kernel):
            """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
            n_channels, _, kw, kh = kernel.shape
            img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
            return F.conv2d(img, kernel, groups=n_channels)

        def laplacian_pyramid(img, kernel, max_levels=5):
            current = img
            pyr = []
            for level in range(max_levels):
                filtered = conv_gauss(current, kernel)
                diff = current - filtered
                pyr.append(diff)
                current = F.avg_pool2d(filtered, 2)
            pyr.append(current)

            return pyr

        if weight is not None:
            logit = logit.clone() * weight
            target = target.clone() * weight

        pyr_logit = laplacian_pyramid(logit, gauss_filter, 5)
        pyr_target = laplacian_pyramid(target, gauss_filter, 5)

        laplacian_loss_weighted = sum(F.l1_loss(a, b) for a, b in zip(pyr_logit, pyr_target))

        return laplacian_loss_weighted

    @staticmethod
    def semantic_loss(pred_trimap, trimap):
        trimap = F.interpolate(trimap, scale_factor=1 / 16, mode='nearest')
        trimap = trimap[:, 0, ...].long()
        semantic_loss = nn.CrossEntropyLoss()(pred_trimap, trimap)

        return semantic_loss

    @staticmethod
    def detail_loss(pred_details, target, gauss_filter, weight=None):
        assert weight is not None, "weight is None!!!"
        pred_os2, pred_os4, pred_os8 = pred_details

        # calculate the alpha loss
        diff2 = (pred_os2 - target)
        edge2_loss = torch.sqrt(diff2 ** 2 + 1e-12) * weight
        # alpha2_loss = torch.sqrt(diff2 ** 2 + 1e-12)

        diff4 = (pred_os4 - target)
        edge4_loss = torch.sqrt(diff4 ** 2 + 1e-12) * weight
        # alpha4_loss = torch.sqrt(diff4 ** 2 + 1e-12)

        diff8 = (pred_os8 - target)
        edge8_loss = torch.sqrt(diff8 ** 2 + 1e-12) * weight
        # alpha8_loss = torch.sqrt(diff8 ** 2 + 1e-12)

        # 归一化
        edge2_loss = edge2_loss.sum() / (weight.sum() + 1)
        edge4_loss = edge4_loss.sum() / (weight.sum() + 1)
        edge8_loss = edge8_loss.sum() / (weight.sum() + 1)

        # weight_all = torch.ones(target.shape).cuda()
        # alpha2_loss = alpha2_loss.sum() / (weight_all.sum() + 1)
        # alpha4_loss = alpha4_loss.sum() / (weight_all.sum() + 1)
        # alpha8_loss = alpha8_loss.sum() / (weight_all.sum() + 1)

        # detail_loss = (edge2_loss + edge4_loss + edge8_loss) + (alpha2_loss + alpha4_loss + alpha8_loss)
        detail_loss = edge2_loss + edge4_loss + edge8_loss

        return detail_loss

    @staticmethod
    def alpha_loss(pred_alpha, target, gauss_filter, weight=None):
        weight_all = torch.ones(target.shape).cuda()

        diff = pred_alpha - target
        # calculate the alpha loss
        alpha_loss = torch.sqrt(diff ** 2 + 1e-12)

        # calculate the edge loss
        edge_loss = alpha_loss * weight

        # 归一化
        alpha_loss = alpha_loss.sum() / (weight_all.sum())
        edge_loss = edge_loss.sum() / (weight.sum())

        fusion_loss = alpha_loss + edge_loss

        return fusion_loss
