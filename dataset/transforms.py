import cv2
import random
import numpy as np


__all__ = ['Compose', 'RandomCrop', 'RandomFlip', 'Normalize']

class Compose:
    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                             'must be equal or larger than 1!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im, label=None):

        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
        if isinstance(label, str):
            label = cv2.imread(label, flags=cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im, label = outputs

        im = im.transpose(2, 0, 1)

        return im, label


class Normalize:
    def __init__(self, mean, var):
        self.mean = np.array(mean)
        self.var = np.array(var)

    def __call__(self, img, label):
        img = img.astype('float32')
        img -= self.mean
        img /= self.var
        return img, label


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label=None):
        if random.random() > self.prob:
            h_flip_img = cv2.flip(img, -1)
            if label is not None:
                h_flip_label = cv2.flip(label, -1)
                return h_flip_img, h_flip_label
            return h_flip_img
        else:
            return img, label


class RandomCrop:
    def __init__(self, prob=0.5, crop_prob=0.5):
        self.prob = prob
        self.crop_prob = crop_prob

    def __call__(self, img, label=None):
        h, w = img.shape[:-1]
        if random.random() > self.prob:
            h_start = random.randint(0, int(self.crop_prob*h))
            w_start = random.randint(0, int(self.crop_prob*w))
            img = img[h_start:h_start + int(self.crop_prob * h), w_start:w_start + int(self.crop_prob * w), :]
            if label is not None:
                label = label[h_start:h_start + int(self.crop_prob * h), w_start:w_start + int(self.crop_prob * w)]
                img, label = cv2.resize(img, dsize=(h, w)), cv2.resize(label, dsize=(h, w))
                return img, label
            img = cv2.resize(img, dsize=(h, w))
            return img
        else:
            return img, label