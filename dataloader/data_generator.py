import cv2
import os
import math
import numbers
import random
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import CONFIG

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.phase = phase

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, trimap = sample['image'][:, :, ::-1], sample['alpha'], sample['trimap']

        # swap color axis
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)

        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1

        if self.phase == "train":
            if "fg" in sample.keys():
                # convert GBR images to RGB
                fg = sample['fg'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
                sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            if "bg" in sample.keys():
                bg = sample['bg'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
                sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)

        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['trimap'] = sample['trimap'][None, ...].float()

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        image, alpha, fg, bg = sample['image'], sample['alpha'], sample['fg'], sample['bg']
        rows, cols, ch = image.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, image.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, image.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        image = cv2.warpAffine(image, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        if fg is not None:
            image = cv2.warpAffine(fg, M, (cols, rows),
                                   flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        if bg is not None:
            alpha = cv2.warpAffine(bg, M, (cols, rows),
                                   flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['image'], sample['alpha'], sample['fg'], sample['bg'] = image, alpha, fg, bg

        return sample

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']

        # if alpha is all 0 skip
        if np.all(alpha == 0):
            return sample

        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        image = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)

        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        image[:, :, 0] = np.remainder(image[:, :, 0].astype(np.float32) + hue_jitter, 360)

        # Saturation noise
        sat_bar = image[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand() * (1.1 - sat_bar) / 5 - (1.1 - sat_bar) / 10
        sat = image[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        image[:, :, 1] = sat

        # Value noise
        val_bar = image[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand() * (1.1 - val_bar) / 5 - (1.1 - val_bar) / 10
        val = image[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        image[:, :, 2] = val

        # convert back to BGR space
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        sample['image'] = image * 255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, alpha, trimap, fg, bg = sample['image'], sample['alpha'], sample['trimap'], sample['fg'], sample['bg']

        if np.random.uniform(0, 1) < self.prob:
            image = cv2.flip(image, 1)
            alpha = cv2.flip(alpha, 1)
            trimap = cv2.flip(trimap, 1)

            if fg is not None:
                fg = cv2.flip(fg, 1)
            if bg is not None:
                bg = cv2.flip(bg, 1)
        sample['image'], sample['alpha'], sample['trimap'], sample['fg'], sample['bg'] = image, alpha, trimap, fg, bg

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, phase="train"):
        self.phase = phase
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        image, alpha, trimap, name = sample['image'], sample['alpha'], sample['trimap'], sample['image_name']
        fg, bg = sample['fg'], sample['bg']

        CROP_SIZE = CONFIG.data.crop_size
        rand_ind = random.randint(0, len(CROP_SIZE) - 1)
        self.output_size = (CROP_SIZE[rand_ind], CROP_SIZE[rand_ind])
        self.margin = self.output_size[0] // 2

        h, w, _ = image.shape
        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w

            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                image = cv2.resize(image, (int(w * ratio), int(h * ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w * ratio), int(h * ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)

                if fg is not None:
                    fg = cv2.resize(fg, (int(w * ratio), int(h * ratio)),
                                    interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                if bg is not None:
                    bg = cv2.resize(bg, (int(w * ratio), int(h * ratio)),
                                    interpolation=maybe_random_interp(cv2.INTER_CUBIC))

                h, w, _ = image.shape

        small_trimap = cv2.resize(trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin // 4:(h - self.margin) // 4,
                                          self.margin // 4:(w - self.margin) // 4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            # self.logger.warning("{} does not have enough unknown area for crop.".format(name))
            left_top = (np.random.randint(0, h - self.output_size[0] + 1),
                        np.random.randint(0, w - self.output_size[1] + 1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0] * 4, unknown_list[idx][1] * 4)

        image_crop = image[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]
        alpha_crop = alpha[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]
        trimap_crop = trimap[left_top[0]:left_top[0] + self.output_size[0],
                      left_top[1]:left_top[1] + self.output_size[1]]

        fg_crop, bg_crop = None, None
        if fg is not None:
            fg_crop = fg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1],
                      :]
        if bg is not None:
            bg_crop = bg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1],
                      :]

        # if there is not enough unknown area for crop
        if len(np.where(trimap == 128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                              "left_top: {}".format(name, left_top))
            image_crop = cv2.resize(image, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)

            if fg is not None:
                fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            if bg is not None:
                bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))

        sample['image'], sample['alpha'], sample['trimap'] = image_crop, alpha_crop, trimap_crop
        sample['fg'], sample['bg'] = fg_crop, bg_crop

        return sample


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, phase='train'):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.phase = phase

    def __call__(self, sample):
        image, alpha, trimap = sample['image'], sample['alpha'], sample['trimap']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if self.phase == 'train':
            if sample['fg'] is not None:
                sample['fg'] = cv2.resize(sample['fg'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if sample['bg'] is not None:
                sample['bg'] = cv2.resize(sample['bg'], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        sample['image'], sample['alpha'], sample['trimap'] = image, alpha, trimap

        return sample


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]
        # sample['origin_trimap'] = sample['trimap']
        # # if h % 32 == 0 and w % 32 == 0:
        # #     return sample
        # # target_h = h - h % 32
        # # target_w = w - w % 32
        # target_h = 32 * ((h - 1) // 32 + 1)
        # target_w = 32 * ((w - 1) // 32 + 1)
        # sample['image'] = cv2.resize(sample['image'], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        # sample['trimap'] = cv2.resize(sample['trimap'], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        if h % 32 == 0 and w % 32 == 0:
            return sample

        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((0, pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

        return sample


class GenTrimap(object):
    def __init__(self):
        kernel_size = random.randint(15, 30)
        self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def __call__(self, sample):
        alpha = sample['alpha']

        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernel)
        bg_mask = cv2.erode(bg_mask, self.erosion_kernel)

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        sample['trimap'] = trimap

        return sample


class DataGenerator(Dataset):
    def __init__(self, data, phase="train", test_scale="origin"):
        self.phase = phase
        self.resize = CONFIG.data.resize
        self.alpha = data.alpha
        if self.phase == "train":
            self.image = data.image
            self.fg = data.fg
            self.bg = data.bg
            self.trimap = None
        else:
            self.image = data.image
            self.trimap = data.trimap

        if CONFIG.data.augmentation:
            train_trans = [
                RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
                GenTrimap(),
                RandomCrop(phase="train"),
                Rescale((self.resize, self.resize), phase='train'),
                RandomJitter(),
                ToTensor(phase="train")]
        else:
            train_trans = [
                GenTrimap(),
                RandomCrop(phase="train"),
                Rescale((self.resize, self.resize), phase='train'),
                ToTensor(phase="train")]

        if test_scale.lower() == "origin":
            test_trans = [OriginScale(), ToTensor()]
        elif test_scale.lower() == "resize":
            test_trans = [Rescale((self.resize, self.resize), phase='test'), ToTensor()]
        elif test_scale.lower() == "crop":
            test_trans = [RandomCrop(phase="test"), ToTensor()]
        else:
            raise NotImplementedError("test_scale {} not implemented".format(test_scale))

        print("test_scale: ", test_scale.lower() == "resize")
        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    Rescale((self.resize, self.resize), phase='test'),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

    def __getitem__(self, idx):
        if self.phase == "train":
            image_path = self.image[idx]
            image = cv2.imread(self.image[idx])
            alpha = cv2.imread(self.alpha[idx], 0).astype(np.float32) / 255

            fg = cv2.imread(self.fg[idx], 1) if self.fg is not None else None
            bg = cv2.imread(self.bg[idx], 1) if self.bg is not None else None

            image_name = os.path.split(self.image[idx])[-1]
            sample = {'image': image, 'alpha': alpha, 'bg': bg, 'fg': fg, 'image_name': image_name}
        else:
            image = cv2.imread(self.image[idx])
            alpha = cv2.imread(self.alpha[idx], 0).astype(np.float32) / 255.

            trimap = None
            if self.trimap[idx] is not None:
                trimap = cv2.imread(self.trimap[idx], 0)

            image_name = os.path.split(self.image[idx])[-1]
            sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'image_name': image_name,
                      'alpha_shape': alpha.shape}

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.alpha)


if __name__ == '__main__':
    from dataloader.image_file import ImageFileTrain, ImageFileTest
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    CONFIG.data.augmentation = False
    # Create a dataloader
    data_dir = "/media/ilab/Innocent/ilab/experiments/datasets/HHM-17k/train"
    # train_image_file = ImageFileTrain(image_dir=data_dir + 'image',
    #                                   alpha_dir=data_dir + "alpha",
    #                                   fg_dir=data_dir + "fg",
    #                                   bg_dir=data_dir + "bg",
    #                                   )
    test_image_file = ImageFileTest(image_dir=CONFIG.data.test_image,
                                    alpha_dir=CONFIG.data.test_alpha,
                                    trimap_dir=CONFIG.data.test_trimap
                                    )

    test_dataset = DataGenerator(test_image_file, phase='test')
    train_dataloader = DataLoader(test_dataset,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=CONFIG.data.workers,
                                 drop_last=False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    import time

    t = time.time()

    b = next(iter(train_dataloader))
    image_name = b['image_name']
    for i in range(b['image'].shape[0]):
        image = b['image'][i]
        image = (image * std + mean).data.numpy() * 255
        image = image.transpose(1, 2, 0)[:, :, ::-1]
        fg = b['fg'][i]
        fg = (fg * std + mean).data.numpy() * 255
        fg = fg.transpose(1, 2, 0)[:, :, ::-1]

        trimap = b['trimap'][i].data.numpy() * 127
        trimap = trimap.transpose(1, 2, 0)

        alpha = b['alpha'][i].data.numpy() * 255
        alpha = alpha.transpose(1, 2, 0)

        cv2.imwrite('../tmp/' + str(i) + '_img.jpg', image.astype(np.uint8))
        cv2.imwrite('../tmp/' + str(i) + '_fg.png', fg.astype(np.uint8))
        cv2.imwrite('../tmp/' + str(i) + '_trimap.png', trimap.astype(np.uint8))
        cv2.imwrite('../tmp/' + str(i) + '_alpha.png', alpha.astype(np.uint8))

        if i > 10:
            break

        print(b['image_name'][i])

        print("image shape:", image.shape)
        print("fg shape:", fg.shape)
        print("trimap shape:", trimap.shape)
        print("alpha shape:", alpha.shape)

    # print(time.time() - t, 'seconds', 'batch_size', batch_size, 'num_workers', num_workers)
