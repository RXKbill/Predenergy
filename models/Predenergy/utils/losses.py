import paddle
import paddle.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Layer):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: paddle.Tensor, freq: int, forecast: paddle.Tensor, target: paddle.Tensor, mask: paddle.Tensor) -> paddle.float32:
        weights = divide_no_nan(mask, target)
        return paddle.mean(paddle.abs((forecast - target) * weights))


class smape_loss(nn.Layer):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: paddle.Tensor, freq: int, forecast: paddle.Tensor, target: paddle.Tensor, mask: paddle.Tensor) -> paddle.float32:
        return 200 * paddle.mean(divide_no_nan(paddle.abs(forecast - target),
                                          paddle.abs(forecast) + paddle.abs(target)) * mask)


class mase_loss(nn.Layer):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: paddle.Tensor, freq: int, forecast: paddle.Tensor, target: paddle.Tensor, mask: paddle.Tensor) -> paddle.float32:
        masep = paddle.mean(paddle.abs(insample[:, freq:] - insample[:, :-freq]), axis=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return paddle.mean(paddle.abs(target - forecast) * masked_masep_inv)
