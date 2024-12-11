from dataclasses import dataclass
from functools import partial
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from criterion.metrics import WeightedAccuracy, Accuracy
from criterion.losses import WeightAcc


@dataclass
class Params:
    seed: int = 42
    layer: str = "DNN" #"DNN" #"LSTM"
    activation: str = "relu" #"relu" "tanh"
    kernel_initializer: str = 'he_normal' #"he_normal" "glorot_uniform"
    input_dim: int = None
    hidden_dim: int = 32
    n_hidden: int = 1
    dropout: float = 0.15
    output_dim: int = 1

    lr: float = 0.001
    l1reg: float = 0.
    opt: str = 'adam'
    criterion: WeightAcc() = WeightAcc()
    gradient_clip: float = None

    batch_size: int = 32
    n_steps: int = 100
    epochs: int = 100
    shuffle: bool = False
    patience_es: int = 5

    metric: WeightedAccuracy = WeightedAccuracy()


class NN:
    """
    This class allows to compile a Neural Network
    """
    def __init__(
            self,
            params: Params
    ):
        self.params = params
        self.set_optimizer()

    def set_optimizer(self):
        if self.params.opt == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=self.params.lr, clipvalue=self.params.gradient_clip)
        else:
            raise ValueError('Only Adam optimizer has been implemented')

    def set_dense(self):
        return partial(layers.Dense, activation=self.params.activation,
                       kernel_initializer=self.params.kernel_initializer,
                       kernel_regularizer=regularizers.l1(self.params.l1reg))

    def set_lstm(self):
        return partial(layers.LSTM, activation=self.params.activation,
                       kernel_initializer=self.params.kernel_initializer,
                       kernel_regularizer=regularizers.l1(self.params.l1reg))

    def set_dropout(self):
        return layers.Dropout(self.params.dropout, seed=self.params.seed)

    def init(self):
        model = models.Sequential()

        if self.params.layer == 'DNN':
            #model.add(layers.Input(shape=[1, self.params.input_dim]))
            model.add(self.set_dense()(self.params.hidden_dim, input_shape=[1, self.params.input_dim]))
            model.add(self.set_dropout())
            for _ in range(self.params.n_hidden):
                model.add(self.set_dense()(self.params.hidden_dim))
                model.add(self.set_dropout())

        elif self.params.layer == 'LSTM':
            #model.add(layers.Input(shape=[self.params.n_steps, self.params.input_dim]))
            model.add(self.set_lstm()(self.params.hidden_dim, input_shape=[self.params.n_steps, self.params.input_dim],
                                       return_sequences=True))
            model.add(self.set_dropout())
            if self.params.n_hidden >= 2:
                for _ in range(1, self.params.n_hidden):
                    model.add(self.set_lstm()(self.params.hidden_dim, return_sequences=True))
                    model.add(self.set_dropout())
            model.add(self.set_lstm()(self.params.hidden_dim))

        model.add(layers.Dense(self.params.output_dim))

        model.compile(optimizer=self.optimizer, loss=self.params.criterion, metrics=[self.params.metric.fn, Accuracy().fn])

        model.summary()
        return model