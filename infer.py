import os
import numpy as np
import cv2
import time

import torch
from torchvision import transforms

from networks import get_generator
from utils import util

MAX_SIZE_H, MAX_SIZE_W = 1600, 1600


def create_model(checkpoint_path):
    # swin_transformer_encoder or res_shortcut_encoder_29
    model = get_generator(encoder="res_shortcut_encoder_29", decoder="res_shortcut_decoder_22")
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(util.remove_prefix_state_dict(ckpt['state_dict']), strict=True)

    return model.cuda()


def transform_image(image, mode="resize", resize=(512, 512)):
    h, w, c = image.shape

    if mode == "resize":
        image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_LINEAR)
    elif mode == "origin":
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    elif mode == "hybrid":
        new_h = min(MAX_SIZE_H, h - (h % 32))
        new_w = min(MAX_SIZE_W, w - (w % 32))
        image = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("mode must be resieze, origin or hybrid!")

    tensor_img = torch.from_numpy(image.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()

    input_t = tensor_img / 255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0)

    return input_t


def inference(model, input_t, mode, origin_size):
    w, h = origin_size

    model.eval()
    with torch.no_grad():
        start = time.time()
        pred_trimap, pred_alpha, _, _, _ = model(input_t)
        end = time.time()
        cost_time = end - start
        pred_alpha = pred_alpha.data.cpu().numpy()[0, 0, :, :]

        pred_trimap = torch.argmax(pred_trimap, dim=1)
        pred_trimap = pred_trimap.data.cpu().numpy()[0, :, :] * 127.

    if mode == "resize" or mode == "hybrid":
        pred_alpha = cv2.resize(pred_alpha, dsize=origin_size, interpolation=cv2.INTER_LINEAR)
        pred_trimap = cv2.resize(pred_trimap, dsize=origin_size)
    elif mode == "origin":
        pred_alpha = pred_alpha[:h, :w]
        pred_trimap = pred_trimap[:h, :w]

    return pred_alpha, cost_time


def main(args):
    model = create_model(args.checkpoint_path)
    model.eval()
    print("create model successfully!!!")
    image_dir = args.image_dir
    pred_dir = args.pred_dir

    names = os.listdir(image_dir)
    count = len(names)
    total_time = 0
    for idx, name in enumerate(names):
        print("Processing: [{}/{}]     Name:{}".format(idx+1, count, name))

        if name[-4:] not in [".png", ".jpg"]:
            count -= 1
            continue

        # read the image
        image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, name)), cv2.COLOR_BGR2RGB)

        # transform the image
        h, w = image.shape[:2]
        input_t = transform_image(image, mode=args.mode)

        pred_alpha, cost_time = inference(model, input_t, mode=args.mode, origin_size=(w, h))
        pred_alpha = pred_alpha * 255
        total_time += cost_time

        # save the image
        save_path = os.path.join(pred_dir, name.replace('.jpg', '.png'))
        cv2.imwrite(save_path, (pred_alpha).astype(np.uint8))

    # print("Average inference time:{:.6f}, fps:{:.1f}".format(total_time / count, count / total_time))


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='/media/ilab/Innocent/ilab/experiments/MyNet'
                                                               '/CFGNet/checkpoints/b1_lap_HHM-17K/best_mse_model.pth')
    # HHM-17K
    parser.add_argument('--image_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/datasets/HHM-17k/test'
                                                         '/image')

    # P3M-10K
    # parser.add_argument('--image_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k'
    #                                                      '/validation/P3M-500-P/blurred_image')

    # AMD
    # parser.add_argument('--image_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/datasets/Composition'
    #                                                      '-1k/human/test/image')

    # real data
    # parser.add_argument('--image_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/datasets/real_test'
    #                                                      '/images')

    parser.add_argument('--pred_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/MyNet/CFGNet'
                                                        '/prediction/debug/resize')

    parser.add_argument('--mode', type=str, default='resize', help="resize || origin || hybrid")

    # Parse configuration
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    main(args)

