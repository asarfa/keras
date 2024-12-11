from keras.src.losses import LossFunctionWrapper
from keras.src.losses.loss import Loss
from .metrics import weighted_acc


class WeightAcc(LossFunctionWrapper):
    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="weighted_acc_loss",
        dtype=None,
    ):
        super().__init__(
            weighted_acc, name=name, reduction=reduction, dtype=dtype
        )

    def get_config(self):
        return Loss.get_config(self)