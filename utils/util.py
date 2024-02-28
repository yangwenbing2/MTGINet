import os
import cv2
import math
import scipy
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils.config import CONFIG


def make_dir(target_dir):
    """
    Create dir if not exists
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def print_network(model, name):
    """
    Print out the network information
    """
    logger = logging.getLogger("Logger")
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()

    logger.info(model)
    logger.info(name)
    logger.info("Number of parameters: {}".format(num_params))


def update_lr(lr, optimizer):
    """
    update learning rates
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr(init_lr, step, iter_num):
    """
    Warm up learning rate
    """
    return step / iter_num * init_lr


def add_prefix_state_dict(state_dict, prefix="module"):
    """
    add prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[prefix + "." + key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    return new_state_dict


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix) + 1:]] = state_dict[key].float()
    return new_state_dict


def load_SwinTransformer_pretrain(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda())
    backbone_state_dict = checkpoint['model']

    model.module.encoder.load_state_dict(backbone_state_dict, strict=False)


def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if CONFIG.model.trimap_channel == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(1).float()

    return weight


def reduce_tensor_dict(tensor_dict, mode='mean'):
    """
    average tensor dict over different GPUs
    """
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            tensor_dict[key] = reduce_tensor(tensor, mode)
    return tensor_dict


def reduce_tensor(tensor, mode='mean'):
    """
    average tensor over different GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= CONFIG.world_size
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt


def get_masked_local_from_global(global_sigmoid, local_sigmoid):
    values, index = torch.max(global_sigmoid, 1)
    index = index[:, None, :, :].float()        # index <===> [0, 1, 2]
    bg_mask = index.clone()     # bg_mask <===> [1, 0, 0]
    bg_mask[bg_mask == 2] = 1
    bg_mask = 1 - bg_mask

    trimap_mask = index.clone()     # trimap_mask <===> [0, 1, 0]
    trimap_mask[trimap_mask == 2] = 0

    fg_mask = index.clone()     # fg_mask <===> [0, 0, 1]
    fg_mask[fg_mask == 1] = 0
    fg_mask[fg_mask == 2] = 1
    
    fusion_sigmoid = local_sigmoid * trimap_mask + fg_mask

    return fusion_sigmoid


Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]


def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    # pred: N, 1 ,H, W
    N, C, H, W = pred.shape
    pred = F.interpolate(pred, size=(640, 640), mode='nearest')
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred < 1.0 / 255.0] = 0
    uncertain_area[pred > 1 - 1.0 / 255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n, 0, :, :]  # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n, 0, :, :] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1

    weight = np.array(weight, dtype=np.float)
    weight = torch.from_numpy(weight).cuda()

    weight = F.interpolate(weight, size=(H, W), mode='nearest')

    return weight


def load_SwinTransformer_pretrain(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda())
    backbone_state_dict = checkpoint['model']

    model.module.encoder.load_state_dict(backbone_state_dict, strict=False)


def load_imagenet_pretrain(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda(CONFIG.gpu))
    state_dict = remove_prefix_state_dict(checkpoint['state_dict'])
    for key, value in state_dict.items():
        state_dict[key] = state_dict[key].float()

    logger = logging.getLogger("Logger")
    logger.debug("Imagenet pretrained keys:")
    logger.debug(state_dict.keys())
    logger.debug("Generator keys:")
    logger.debug(model.module.encoder.state_dict().keys())
    logger.debug("Intersection  keys:")
    logger.debug(set(model.module.encoder.state_dict().keys()) & set(state_dict.keys()))

    weight_u = state_dict["conv1.module.weight_u"]
    weight_v = state_dict["conv1.module.weight_v"]
    weight_bar = state_dict["conv1.module.weight_bar"]

    logger.debug("weight_v: {}".format(weight_v))
    logger.debug("weight_bar: {}".format(weight_bar.view(32, -1)))
    logger.debug("sigma: {}".format(weight_u.dot(weight_bar.view(32, -1).mv(weight_v))))

    new_weight_v = torch.zeros(3, 3, 3).cuda()
    new_weight_bar = torch.zeros(32, 3, 3, 3).cuda()

    new_weight_v[:3, :, :].copy_(weight_v.view(3, 3, 3))
    new_weight_bar[:, :3, :, :].copy_(weight_bar)

    logger.debug("new weight_v: {}".format(new_weight_v.view(-1)))
    logger.debug("new weight_bar: {}".format(new_weight_bar.view(32, -1)))
    logger.debug("new sigma: {}".format(weight_u.dot(new_weight_bar.view(32, -1).mv(new_weight_v.view(-1)))))

    state_dict["conv1.module.weight_v"] = new_weight_v.view(-1)
    state_dict["conv1.module.weight_bar"] = new_weight_bar

    model.module.encoder.load_state_dict(state_dict, strict=False)
