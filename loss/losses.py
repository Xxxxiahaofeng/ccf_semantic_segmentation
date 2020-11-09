import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        weight (list): the weight of every class in loss function. Default ``None``.
    """

    def __init__(self, ignore_index=255, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss = paddle.nn.loss.CrossEntropyLoss(weight=paddle.to_tensor(weight), ignore_index=ignore_index)

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        avg_loss = self.loss(logit, label)
        return avg_loss