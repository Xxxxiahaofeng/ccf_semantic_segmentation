import os
import paddle
import logging
import numpy as np

LEVEL = {'DEBUG':logging.DEBUG,
         'INFO':logging.INFO,
         'WARNING':logging.WARNING,
         'ERROR':logging.ERROR,
         'CRITICAL':logging.CRITICAL}


def logger(path, name, level='INFO', if_cover=True):
    logger = logging.getLogger(name=name)
    logger.setLevel(LEVEL[level])
    if if_cover:
        file = logging.FileHandler(filename=os.path.join(path, 'log.txt'), mode='w')
    else:
        file = logging.FileHandler(filename=os.path.join(path, 'log.txt'))
    file.setLevel(LEVEL[level])

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(LEVEL[level])

    logger.addHandler(file)
    logger.addHandler(console)

    return logger


def load_pretrained_model(model, pretrained_model):
    logger = logging.getLogger('train.load_pretrained')
    if os.path.exists(pretrained_model):
        para_state_dict = paddle.load(pretrained_model)

        model_state_dict = model.state_dict()
        keys = model_state_dict.keys()
        num_params_loaded = 0
        for k in keys:
            k_para = k.replace('backbone.', '')
            if k_para not in para_state_dict:
                logger.warning("{} is not in pretrained model".format(k))
            elif list(para_state_dict[k_para].shape) != list(
                    model_state_dict[k].shape):
                logger.warning(
                    "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k_para].shape,
                                model_state_dict[k].shape))
            else:
                model_state_dict[k] = para_state_dict[k_para]
                num_params_loaded += 1
        model.set_dict(model_state_dict)
        logger.info("There are {}/{} variables loaded into {}.".format(
            num_params_loaded, len(model_state_dict),
            model.__class__.__name__))
    else:
        raise ValueError(
            'The pretrained model directory is not Found: {}'.format(
                pretrained_model))


def load_resume_model(model, dir, resume):
    logger = logging.getLogger('train.load_resume')
    resume_model = os.path.join(dir, '{}_epoch.pdparams'.format(resume))
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model)
            para_state_dict = paddle.load(ckpt_path)
            model.set_dict(para_state_dict)
            epoch = resume_model.split('_')[-1]
            if epoch.isdigit():
                epoch = int(epoch)
            return epoch
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')


class runningScore(object):

    def __init__(self, n_classes=11):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IoU
            - F1
            - recall
        """
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / (hist.sum() + 1e-6)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-6)
        # print(acc_cls[1])
        recall = np.diag(hist) / hist.sum(axis=0)
        # mean_cls = np.nanmean(acc_cls)c

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-6)
        mean_iu = np.nanmean(iu)
        # freq = hist.sum(axis=1) / hist.sum()
        # # print(freq)
        # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        F1 = 2 * acc_cls[1] * recall[1] / (acc_cls[1] + recall[1] + 1e-6)
        return {'Overall Acc: \t': acc,
                'Class Acc: \t': acc_cls,
                'Recall: \t': recall,
                'Class IoU: \t': cls_iu,
                'F1 Score: \t': F1,
                'Mean IoU : \t': mean_iu}
        # return {'Overall Acc: \t': acc,
        #         'Mean Acc : \t': mean_cls,
        #         'Class Acc : \t': acc_cls,
        #         'FreqW Acc : \t': fwavacc,
        #         'Mean IoU : \t': mean_iu,}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class warmup_lr:
    def __init__(self, start_lr, end_lr, num_step, last_step=-1):
        self.num_step = num_step
        self.last_step = last_step
        self.now_step = last_step + 1

        self.step_lr = (end_lr - start_lr)/num_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def set(self, optimizer):
        if self.now_step < self.num_step:
            optimizer.set_lr(self.start_lr+self.now_step*self.step_lr)
            self.now_step += 1
        return optimizer


COLOR_TABLE = [[96,128,0],
               [64,128,0],
               [0,128,64],
               [96,128,192],
               [32,128,192],
               [64,128,64],
               [96,64,64],
               [255,255,255]]

def label2color(label):
    color_map = np.zeros((label.shape[0], label.shape[1], 3))
    for id in range(len(COLOR_TABLE)):
        color_map[label == id, 0] = COLOR_TABLE[id][2]
        color_map[label == id, 1] = COLOR_TABLE[id][1]
        color_map[label == id, 2] = COLOR_TABLE[id][0]
    return color_map