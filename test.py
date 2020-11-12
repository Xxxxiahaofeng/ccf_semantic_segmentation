import cv2
import paddle
import argparse
import numpy as np
from tqdm import tqdm

from utils import *
from dataset import baseDataset
from dataset.transforms import *
from models.build import build_model
from paddle.io import Dataset, DataLoader


class tester:
    def __init__(self, args):
        self.args = args
        self.n_class = args.n_class
        self.models = []

        for i, model_name in enumerate(args.models):
            model = build_model(model_name, args.n_class)
            load_resume_model(model, args.resumedir, args.resumes[i])
            model.eval()
            self.models.append(model)

        test_dataset = baseDataset(dir=args.datadir,
                                   num_class=args.n_class,
                                   split='test')
        self.datalist = test_dataset.datalist
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            return_list=True
        )

    def test(self):
        for meta in tqdm(self.test_loader):
            img = meta[0]
            img_id =meta[1]
            img_names = [self.datalist[i][self.datalist[i].rindex('/')+1:] for i in img_id]
            preds = np.zeros((img.shape[0], self.args.n_class, img.shape[2], img.shape[3]))

            for model in self.models:
                logits = model(img)
                pred_i = paddle.nn.functional.softmax(logits, axis=1)
                pred_i = pred_i.numpy().astype('float32')
                preds += np.squeeze(pred_i)

            preds = np.argmax(preds/len(self.models), axis=1)

            for i, name_i in enumerate(img_names):
                pred_i = preds[i,:,:]
                cv2.imwrite(os.path.join(self.args.outdir,name_i.replace('jpg', 'png')), pred_i)


def main(args):
    mytester = tester(args)
    mytester.test()

if __name__ == "__main__":
    paddle.disable_static()
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/media/xiehaofeng/新加卷/things/baidu_segmentation/img_testA', help="the path of data")
    parser.add_argument('--outdir', default='/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/result/test_result', help='the path to save checkpoints, val_result and log file')
    parser.add_argument('--resumedir', default='/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/result/resume_models')
    parser.add_argument('--resumes', default=[13, 29], help='the resume parameters of models, should equal to the number of models')

    parser.add_argument('--n_class', default=7, help='the number of classes')
    parser.add_argument('--batch_size', default=16, help='the number of batch size')
    parser.add_argument('--ignore_value', default=255, help='the label be ignore in training')

    parser.add_argument('--models', default=['HRNet_W48_OCR', 'UNet'], help='the model to be build and train')
    args = parser.parse_args()
    main(args)