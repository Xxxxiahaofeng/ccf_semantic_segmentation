import os
import cv2
import paddle
import numpy as np
import paddle.fluid.io as io


class baseDataset(io.Dataset):
    def __init__(self, dir, num_class, transform=None, split='train', ignore_index=255):
        super(baseDataset, self).__init__()

        assert os.path.isfile(os.path.join(dir, '{}.txt'.format(split))), \
            '{} has no dataset file {}.txt.'.format(dir, split)
        with open(os.path.join(dir, '{}.txt'.format(split))) as f:
            self.datalist = f.readlines()

        self.transform = transform
        self.num_class = num_class
        self.split = split
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, item):
        mean = np.array([72.57055382, 90.68219696, 81.40952313])
        var = np.array([51.17250644, 53.22876833, 60.39464522])

        if self.split == 'train':
            imgdir, labeldir = self.datalist[item].replace('\n', '').split(' ')
            img = cv2.imread(imgdir, flags=cv2.IMREAD_COLOR)
            label = cv2.imread(labeldir, flags=cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                img, label = self.transform(img, label)
            label[label >= self.num_class] = 255
            label[label < 0] = 255
            return img, label.astype('int64')

        elif self.split == 'val':
            imgdir, labeldir = self.datalist[item].replace('\n', '').split(' ')
            img = cv2.imread(imgdir, flags=cv2.IMREAD_COLOR)
            label = cv2.imread(labeldir, flags=cv2.IMREAD_GRAYSCALE)

            img = img.astype('float32')
            img -= mean
            img /= var
            img = img.transpose(2, 0, 1)

            return img, label.astype('int64')

        elif self.split == 'test':
            imgdir = self.datalist[item].replace('\n', '')
            img = cv2.imread(imgdir, flags=cv2.IMREAD_COLOR)

            img = img.astype('float32')
            img -= mean
            img /= var
            img = img.transpose(2, 0, 1)

            return img, item
