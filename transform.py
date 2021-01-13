import random
import math
import numpy as np
import numbers
import collections
import cv2

import torch


class Compose(object):
    """
        Example:
            transform.Compose(
                [
                 transform.RandScale([0.5, 2.0]),
                 transform.ToTensor()
                 ]
            )
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, label):
        for t in self.transform:
            image, label = t(image, label)
        return image, label


class ToTensor(object):
    """
        1. Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
        2. Not div 255
        3. output: image-type:float, label-type: long
    """
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):    #
            raise (RuntimeError("transform.ToTensor() only handle np.ndarray [eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("transform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("transform.ToTensor() only handle np.ndarray label with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))    # image shape : C * H * W
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)                         # label shape : H * W
        if not isinstance(label, torch.LongTensor):             # 这样安排label和image的shape是为了交叉熵方便
            label = label.long()
        return image, label


class Normalize(object):
    """
        Normalize tensor with mean and standard deviation along channel:
            channel = (channel - mean) / std
    """
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label


class Resize(object):
    """
        Resize the input to the given size, 'size' is a 2-element tuple in the order of (h, w)
    """

    def __init__(self, size: tuple):
        assert len(size) == 2
        self.size = size

    def __call__(self, image, label):
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label


class RandScale(object):
    """
        Randomly resize image & label with scale factor in [scale_min, scale_max]
        scale: 图片等比例放大的系数, aspect_ratio: 宽高的比例系数
        p: The probability of random scale operation. The default value is 0.5
    """
    def __init__(self, scale: list, aspect_ratio=None, p=0.5):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("transform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("transform.RandScale() aspect_ratio param error.\n"))
        if isinstance(p, numbers.Number):
            self.p = p
        else:
            raise (RuntimeError("probability: p should be a Number range in [0,1].\n"))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None and random.random() > self.p:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        1. size (sequence or int): output size of the Crop operation.
            If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
        2. output-size can greater than the input-size, at this time you should give the padding to fill the more space,
            and padding should be a list with len() == 3, because the number of RGB channel is 3
        3. ignore_label is the Pixel value used in padding operation when size greater than input-size
        4. the crop_type work only when the output-size less than input-size, and value can be rand or center
           when center means Clipping range: H: [h/2 - crop_h/2, h/2 + crop_h/2], W: [w/2 - crop_w/2, w/2 + crop_w/2]
    """
    def __init__(self, size: list, crop_type='center', padding=None, ignore_label=255):
        if padding is None:
            padding = [255, 255, 255]

        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3 because RGB image has 3 channel\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            # 给图片四周加上边框，以实现Crop的结果比原图还要大的要求
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label


class RandRotate(object):
    """
        Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]

        rotate: [rotate_min, rotate_max]
        padding default is [255, 255, 255], in other words image padding fill with black pixel
        ignore_label default is 0, in other words label padding fill with black pixel
        p: Random probability, should be a float number in [0, 1]
    """
    def __init__(self, rotate, padding=None, ignore_label=255, p=0.5):
        if padding is None:
            padding = [255, 255, 255]
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("transform.RandRotate() scale param error.\n"))
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    """
        function: HorizontalFlip
        p: Random probability, should be a float number in [0, 1]
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class RGB2BGR(object):
    # Converts image from RGB order to BGR order
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label