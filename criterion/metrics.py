import tensorflow as tf
from keras.src import ops
from keras.src.metrics import reduction_metrics, Metric
from keras.src.losses.loss import squeeze_or_expand_to_same_rank


def acc(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    dir_true = tf.where(y_true > 0, 1, -1)
    dir_pred = tf.where(y_pred > 0, 1, -1)
    return tf.reduce_sum(tf.where(dir_true*dir_pred == 1, 1., 0.))/tf.reduce_sum(tf.where(y_true>0, 1., 1.))


class Accuracy:
    def __init__(self, name="accuracy"):
        self.name = name
        self.fn = acc
        self._direction = "max"


def weighted_acc(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    dir_true = tf.where(y_true > 0, 1, -1)
    dir_pred = tf.where(y_pred > 0, 1, -1)
    indic = tf.where(dir_true*dir_pred == 1, 1., 0.)
    abs_dev = tf.abs(y_true - y_pred)
    return tf.reduce_sum(abs_dev * indic) / (tf.reduce_sum(abs_dev) + 1e-10)


class WeightedAccuracy:
    def __init__(self, name="weighted_acc"):
        self.name = name
        self.fn = weighted_acc
        self._direction = "max"

"""
class WeightedAccuracy(reduction_metrics.MeanMetricWrapper):
    def __init__(self, name="weighted_acc", dtype=None):
        super().__init__(fn=weighted_acc, name=name, dtype=dtype)
        self._direction = "max"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}
        
class WeightedAccuracy(Metric):
    def __init__(self, name="weighted_acc", **kwargs):
        super().__init__(**kwargs)  # handles base args (e.g., dtype)
        self.name = name
        self.fn = weighted_acc
        self._direction = 'max'
        self.wacc = self.add_weight(name='wacc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.wacc.assign(self.fn(y_true, y_pred))

    def result(self):
        return self.wacc

    def reset_state(self):
        self.wacc.assign(0)
"""