# import os
# imgdir = '/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/img_train'
# labeldir = '/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/label'
#
# imgs = os.listdir(imgdir)
# labels = os.listdir(labeldir)
# trainlist = open('/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/train.txt', 'w+')
#
# for img in imgs:
#     line = '{}/{} {}/{}\n'.format(imgdir, img, labeldir, img.replace('jpg', 'png'))
#     if os.path.isfile(os.path.join(labeldir, img.replace('jpg', 'png'))):
#         trainlist.write(line)

# ======================================================================================================================
# import os
# import cv2
# import numpy as np
# import multiprocessing as mp
# import multiprocessing.pool as mpp
#
#
# color_table = [[96,128,0],
#                [64,128,0],
#                [0,128,64],
#                [96,128,192],
#                [32,128,192],
#                [64,128,64],
#                [96,64,64],
#                [255,255,255]]
#
# labeldir = 'E:\\things\\baidu_segmentation\\train_data\\color_label'
# savedir = 'E:\\things\\baidu_segmentation\\train_data\\label'
#
# def process(labelname):
#     color_label = cv2.imread(os.path.join(labeldir, labelname))
#     color_label = cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB)
#     label = np.zeros(color_label.shape[:-1])
#     for i, color in enumerate(color_table):
#         index = np.all(color_label == color, axis=2)
#         label[index] = i
#     label[label == 7] = 255
#     cv2.imwrite(os.path.join(savedir, labelname), label)
#
#
# if __name__ == '__main__':
#     labellist = os.listdir(labeldir)
#     mpp.Pool(processes=mp.cpu_count()).map(process, labellist)

# ======================================================================================================================

import os
import cv2
import random
from tqdm import tqdm
import numpy as np

labeldir = '/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/label'
labelfiles = os.listdir(labeldir)
hist_list = np.zeros((7, len(labelfiles)))
i=0

# for labelfile in tqdm(labelfiles):
#     label = cv2.imread(filename=os.path.join(labeldir, labelfile), flags=cv2.IMREAD_GRAYSCALE)
#     hist = np.histogram(label, bins=7, range=(0,7))
#     hist_list[:, i] = hist[0]/np.sum(hist[0])
#     i+=1
#
# sorted_labelfile_id = np.zeros((7, len(labelfiles)))
# for index in range(7):
#     file_id = np.array([range(len(labelfiles))]).astype(int)
#     concated = np.concatenate((file_id, [hist_list[index, :]]), axis=0)
#     concated = concated[:, concated[1, :].argsort()][:, ::-1]
#     sorted_labelfile_id[index] = concated[0, :]
#
# sorted_labelfile_id = sorted_labelfile_id.astype(int)
# np.save("sorted_label_id.npy", sorted_labelfile_id)

sorted_labelfile_id = np.load('sorted_label_id.npy').astype(int)

sorted_labelfile_id = sorted_labelfile_id[:, 0:3000].flatten()
sorted_labelfile_id = list(set(sorted_labelfile_id))
index = sorted(random.sample(sorted_labelfile_id, 5000))
val_labelfiles = [labelfiles[x] for x in index]
train_labelfiles = list(set(labelfiles) - set(val_labelfiles))

all_hist = np.zeros((7,1))
for labelfile in tqdm(val_labelfiles):
    label = cv2.imread(filename=os.path.join(labeldir, labelfile), flags=cv2.IMREAD_GRAYSCALE)
    hist = np.histogram(label, bins=7, range=(0,7))
    all_hist[:, 0] += hist[0]/np.sum(hist[0])
    i+=1
print(all_hist/i)

imgdir = '/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/img_train'
labeldir = '/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/label'
trainlist = open('/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/train.txt', 'w+')
vallist = open('/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/val.txt', 'w+')

for label in tqdm(train_labelfiles):
    line = '{}/{} {}/{}\n'.format(imgdir, label.replace('png', 'jpg'), labeldir, label)
    if os.path.isfile(os.path.join(imgdir, label.replace('png', 'jpg'))):
        trainlist.write(line)

for label in tqdm(val_labelfiles):
    line = '{}/{} {}/{}\n'.format(imgdir, label.replace('png', 'jpg'), labeldir, label)
    if os.path.isfile(os.path.join(imgdir, label.replace('png', 'jpg'))):
        vallist.write(line)