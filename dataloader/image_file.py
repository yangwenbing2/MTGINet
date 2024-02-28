import os
import glob
import logging
import functools
import numpy as np


class ImageFile(object):
    def __init__(self, phase='train'):
        self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names
        name_sets = [self._get_name_set(d) for d in dirs]

        # Reduce
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        if len(valid_names) == 0:
            self.logger.error('No image valid')
        else:
            self.logger.info('{}: {} foreground/images are valid'.format(self.phase.upper(), len(valid_names)))

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                 image_dir="train_image",
                 alpha_dir="train_alpha",
                 fg_dir=None,
                 bg_dir=None,
                 image_ext=".jpg",
                 alpha_ext=".png",
                 fg_ext=".png",
                 bg_ext=".png",
                 # image_ext=".png",
                 # alpha_ext=".jpg",
                 # fg_ext=".jpg",
                 # bg_ext=".png",
                 ):
        super(ImageFileTrain, self).__init__(phase="train")

        self.image_dir = image_dir
        self.alpha_dir = alpha_dir
        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.image_ext = image_ext
        self.alpha_ext = alpha_ext
        self.fg_ext = fg_ext
        self.bg_ext = bg_ext

        self.logger.debug('Load Training Images From Folders')

        self.valid_alpha_list = self._get_valid_names(self.image_dir, self.alpha_dir)

        self.image = self._list_abspath(self.image_dir, self.image_ext, self.valid_alpha_list)
        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_alpha_list)

        self.fg, self.bg = None, None
        if fg_dir is not None:
            self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_alpha_list)

        if bg_dir is not None:
            self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_alpha_list)

    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 image_dir="test_image",
                 alpha_dir="test_alpha",
                 trimap_dir=None,
                 image_ext=".jpg",
                 alpha_ext=".png",
                 trimap_ext=".png",
                 # image_ext=".png",
                 # alpha_ext=".png",
                 # trimap_ext=".png",
                 ):
        super(ImageFileTest, self).__init__(phase="test")

        self.image_dir = image_dir
        self.alpha_dir = alpha_dir
        self.trimap_dir = trimap_dir
        self.image_ext = image_ext
        self.alpha_ext = alpha_ext
        self.trimap_ext = trimap_ext

        self.logger.debug('Load Testing Images From Folders')

        self.valid_image_list = self._get_valid_names(self.image_dir, self.alpha_dir, shuffle=False)

        self.image = self._list_abspath(self.image_dir, self.image_ext, self.valid_image_list)
        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)

        self.trimap = None
        if self.trimap_dir is not None:
            self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S')

    train_data = ImageFileTrain(
        image_dir="/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k/train/blurred_image",
        alpha_dir="/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k/train/mask",
        fg_dir="/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k/train/fg",
        bg_dir="/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k/train/bg"
    )
    test_data = ImageFileTest(
        image_dir="/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k/validation/P3M-500-P/blurred_image",
        alpha_dir="/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k/validation/P3M-500-P/mask",
        trimap_dir="/media/ilab/Innocent/ilab/experiments/datasets/P3M-10k/validation/P3M-500-P/trimap"
    )

    print("Train files:")
    print("image:", train_data.image[0])
    print("alpha:", train_data.alpha[0])
    print("image and alpha:", len(train_data.image), len(train_data.alpha))

    if train_data.fg is not None:
        print("fg:", train_data.fg[0])
        print(len(train_data.fg))
    if train_data.bg is not None:
        print("bg:", train_data.bg[0])
        print(len(train_data.bg))

    print("Test files:")
    print("image:", test_data.image[0])
    print("alpha:", test_data.alpha[0])
    print("trimap:", test_data.trimap[0])
    print(len(test_data.image), len(test_data.alpha), len(test_data.trimap))
