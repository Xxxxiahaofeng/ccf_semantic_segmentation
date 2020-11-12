import os
import cv2
import time
import visdom
import paddle
import argparse
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.io import Dataset, DataLoader

from utils import *
from dataset import baseDataset
from dataset.transforms import *
from models.build import build_model
from loss.build import build_loss

vis = visdom.Visdom()


class trainer:
    def __init__(self, args):
        if not os.path.isdir(args.outdir):
            os.mkdir(args.outdir)
        self.args = args
        self.logger = logger(args.outdir, name='train')
# ==================== model, optimizer, loss ====================
        self.model = build_model(args.model, args.n_class)
        self.optimizer = paddle.optimizer.Adam(beta1=0.9,
                                               beta2=0.999,
                                               learning_rate=args.lr,
                                               weight_decay=args.weight_decay,
                                               parameters=self.model.parameters())
        self.warmup_lr = warmup_lr(start_lr=0.0001,
                                   end_lr=args.lr,
                                   num_step=5,
                                   last_step=self.args.resume)

        if args.resume >= 0:
            load_resume_model(self.model, args.outdir, args.resume)
        elif args.pretrained is not None:
            load_pretrained_model(self.model, args.pretrained)

        if dist.get_world_size() > 1:
            self.logger.info('training on {} gpus'.format(dist.get_world_size()))
            fleet.init(is_collective=True)
            strategy = fleet.DistributedStrategy()
            self.optimizer = fleet.distributed_optimizer(self.optimizer, strategy=strategy)
            self.model = fleet.distributed_model(self.model)

        self.loss = build_loss(args)
# ==================== transform, data loader ====================
        transforms = Compose(transforms=[RandomCrop(),
                                         RandomFlip(),
                                         Normalize([72.57055382, 90.68219696, 81.40952313],
                                                   [51.17250644, 53.22876833, 60.39464522])
                                         ])
        train_dataset = baseDataset(dir=args.datadir,
                              transform=transforms,
                              num_class=args.n_class)
        eval_dataset = baseDataset(dir=args.datadir,
                                   num_class=args.n_class,
                                   split='val')

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            return_list=True,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            return_list=True
        )
# ==================== other settings ====================
        self.n_epoch = args.n_epoch
        self.maxiter = args.maxiter
        self.resume = args.resume
        self.num_batches = int(len(train_dataset)/args.batch_size)
        if args.resume >= 5:
            self.start_iter = int(self.num_batches)*(args.resume - 4)
        else:
            self.start_iter = 0
        self.running_metrics = runningScore(n_classes=args.n_class)

    def train(self):
        all_train_iter_loss = []
        all_val_epo_iou = []

        curr_iter = self.start_iter
        metric_writer = open(os.path.join(self.args.outdir, 'eval_metric.txt'), 'w+')

        for epoch_i in range(self.resume+1, self.n_epoch):

            iter_loss = AverageTracker()
            train_loss = AverageTracker()
            data_time = AverageTracker()
            batch_time = AverageTracker()
            tic = time.time()
# ==================== train ======================
            self.model.train()

            if self.warmup_lr.now_step < self.warmup_lr.num_step and self.args.if_warmup:
                warmingup = True
                self.optimizer = self.warmup_lr.set(self.optimizer)
            else:
                warmingup = False

            for i, meta in enumerate(self.train_loader):
                images = meta[0].astype('float32')
                labels = meta[1].astype('int64')
                data_time.update(time.time() - tic)

                logits = self.model(images)
                loss = self.loss(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.model.clear_gradients()

                if not warmingup and self.args.lr_decay:
                    self.optimizer.set_lr(self.args.lr*(1-float(curr_iter)/(self.num_batches*self.n_epoch))**0.9)
                    curr_iter += 1

                iter_loss.update(loss.numpy()[0])
                train_loss.update(loss.numpy()[0])
                batch_time.update(time.time() - tic)
                tic = time.time()

                if i % 10 == 0:
                    all_train_iter_loss.append(iter_loss.avg)
                    iter_loss.reset()
                    vis.line(all_train_iter_loss, win='train iter loss', opts=dict(title='train iter loss'))

                log = 'Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr: {:.6f}, Loss: {:.6f}' \
                    .format(epoch_i, i, self.num_batches,
                            batch_time.avg, data_time.avg,
                            self.optimizer.get_lr(),
                            loss.numpy()[0],)
                self.logger.info(log)
# ==================== eval ======================
            self.eval(epoch_i)
            score = self.running_metrics.get_scores()
            oa = score['Overall Acc: \t']
            acc = score['Class Acc: \t'][1]
            recall = score['Recall: \t'][1]
            iou = score['Class IoU: \t'][1]
            F1 = score['F1 Score: \t']
            miou = score['Mean IoU : \t']
            self.running_metrics.reset()

            all_val_epo_iou.append(miou)
            vis.line(all_val_epo_iou, win='val epoch iou', opts=dict(title='val epoch iou'))

            epoch_log = 'Epoch Val: [{}], ACC: {:.2f}, Recall: {:.2f}, meanIoU: {:.6f}' \
                .format(epoch_i, acc, recall, miou)
            self.logger.info(epoch_log)
            metric_writer.write(epoch_log+'\n')

            paddle.save(self.model.state_dict(),
                        os.path.join(self.args.outdir, '{}_epoch.pdparams'.format(epoch_i)))

    def eval(self, epoch_i):
        self.model.eval()
        for i, meta in enumerate(self.eval_loader):

            images = meta[0]
            labels = meta[1]

            logits = self.model(images)
            pred = paddle.argmax(logits, axis=1)

            pred = pred.numpy().astype(int)
            labels = labels.numpy().astype(int)
            pred = np.squeeze(pred)
            labels = np.squeeze(labels)

            self.running_metrics.update(labels, pred)

            if i % 2000 == 0:
                color_pred = label2color(pred[0,:,:])
                cv2.imwrite(os.path.join(self.args.outdir, 'val_pred', 'epoch_{}_{}.png'.format(epoch_i, i)), color_pred)



def main(args):
    mytrainer = trainer(args)
    mytrainer.train()


if __name__ == "__main__":
    paddle.disable_static()
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data', help="the path of data")
    parser.add_argument('--pretrained', default='/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/result/resnet101_vd_ssld/model.pdparams', help='the path of pretrained weight')
    parser.add_argument('--outdir', default='/media/xiehaofeng/新加卷/things/baidu_segmentation/train_data/result/deeplab_checkpoint', help='the path to save checkpoints, val_result and log file')

    parser.add_argument('--n_epoch', default=50, help='train epoch numbers')
    parser.add_argument('--resume', default=0, help='the epoch to start train')
    parser.add_argument('--maxiter', default=50000, help='the max iteration of training')

    parser.add_argument('--n_class', default=7, help='the number of classes')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--if_warmup', default=False, help='if using warmup learning rate')
    parser.add_argument('--lr_decay', default=True, help='if use learning rate dacay')
    parser.add_argument('--batch_size', default=12, help='the number of batch size')
    parser.add_argument('--ignore_value', default=255, help='the label be ignore in training')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD optimizer')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='waight decay of SGD optimizer')

    parser.add_argument('--model', default='deeplabv3p', help='the model to be build and train')

    parser.add_argument('--loss', default='CrossEntropyLoss', help='the loss using for train and validate')
    parser.add_argument('--loss_weight', default=[1.0,1.0,0.5,1.0,10.0,1.0,1.0])

    args = parser.parse_args()
    main(args)