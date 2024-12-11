import numpy as np
import pandas as pd
from .baseline import Params, NN
from dataset_engineering.env import Env
from keras import callbacks
import os
from criterion.metrics import acc
import tensorflow as tf


class Model(NN):
    def __init__(
            self,
            train_env: Env = None,
            val_env: Env = None,
            params: Params = None,
            project_name: str = None,
            nn_hp=None,
            verbose: bool = False,
            vis_board: bool = False
    ):
        super().__init__(params=params)
        self.train_env = train_env
        self.val_env = val_env
        self.project_name = project_name
        self.nn_hp = nn_hp
        self.verbose = verbose
        self.vis_board = vis_board
        self.set_name()
        self.set_dir()
        self.set_model()
        self.set_monitor()

    def set_name(self):
        params = {key: value for key, value in sorted(self.params.__dict__.items())}
        keys = self.project_name.split("__")
        self.name = "__".join([str(k) + '_' + str(round(v, 4)) for k, v in params.items() if k in keys])

    def set_dir(self):
        if self.nn_hp is None:
            self.dirpath = f'logs/{self.project_name}/{self.name}'
        else:
            self.dirpath = f'hp_logs/{self.project_name}/{self.name}'

    def set_model(self):
        if self.nn_hp is None:
            self.model = self.init()
        else:
            self.model = self.nn_hp
            self.nn_hp = True

    def set_monitor(self):
        if self.val_env is not None:
            self.monitor = f'val_{self.params.metric.name}'
        else:
            self.monitor = f'{self.params.metric.name}'

    def checkpoint(self):
        return callbacks.ModelCheckpoint(monitor=self.monitor, mode=self.params.metric._direction,
                                         save_weights_only=True,
                                         filepath=f'{self.dirpath}checkpoint.weights.h5',
                                         save_best_only=True)

    def earlystopping(self):
        return callbacks.EarlyStopping(monitor=self.monitor, mode=self.params.metric._direction,
                                       patience=self.params.patience_es, restore_best_weights=True)

    def scheduler(self):
        return callbacks.ReduceLROnPlateau(monitor=self.monitor, mode=self.params.metric._direction,
                                           factor=0.2, patience=int(self.params.patience_es * 0.6))

    def tensoarboard(self):
        return callbacks.TensorBoard(self.dirpath)

    @staticmethod
    def find_data(layer: str, env: Env):
        if layer == 'DNN':
            return env.X_tab, env.y_tab
        elif layer == 'LSTM':
            return env.X_ts, env.y_ts

    def init_data(self):
        X_train, y_train = self.find_data(self.params.layer, self.train_env)
        if self.val_env is not None:
            X_val, y_val = self.find_data(self.params.layer, self.val_env)
        else:
            X_val, y_val = None, None
        return X_train, y_train, X_val, y_val

    def fit(self, tune_callbacks=None):
        X_train, y_train, X_val, y_val = self.init_data()
        validation_data = None if self.val_env is None else (X_val, y_val)
        callbacks = [self.earlystopping(), self.scheduler()]
        if self.nn_hp is None: callbacks.append(self.checkpoint())
        if self.vis_board: callbacks.append(self.tensoarboard())
        if tune_callbacks is not None: callbacks.extend(tune_callbacks)
        return self.model.fit(X_train, y_train, batch_size=self.params.batch_size,
                              epochs=self.params.epochs, validation_data=validation_data,
                              callbacks=callbacks, shuffle=self.params.shuffle,
                              validation_batch_size=self.params.batch_size, verbose=self.verbose)

    def load_weights(self):
        self.model.load_weights(self.dirpath)

    def predict(self, X: np.array):
        return self.model.predict(X, batch_size=self.params.batch_size)

    def evaluate(self, env):
        X, y = self.find_data(self.params.layer, env)
        return self.model.evaluate(X, y, batch_size=self.params.batch_size, return_dict=True)

    def infer_predict(self, env: Env, save: bool = False):
        X, y = self.find_data(self.params.layer, env)
        outputs = self.predict(X)
        if self.params.layer == "DNN":
            outputs = tf.squeeze(outputs, axis=2)
        if y is not None:
            if env.scale_target:
                outputs = env.scaler_y.inverse_transform(outputs)
                y = self.reshape_raw_y(env, y)
            #self.params.metric.reset_state()
            outputs = tf.cast(outputs, tf.float32)
            metric = self.params.metric.fn(y, outputs).numpy()
            print(f'val_{self.params.metric.name}=', round(metric, 6))
            mode = 'val'
            accuracy = acc(y, outputs).numpy()
            print(f'val_acc=', round(accuracy, 6))
        else:
            if env.scale_target:
                outputs = env.scaler_y.inverse_transform(outputs)
            mode = 'test'
        if save:
            filename = f'outputs/{self.project_name}/{mode}/{self.name}.csv'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pd.Series(outputs.squeeze(), index=env.data.index[-len(outputs):]).to_csv(filename)
        if mode =='val':
            return outputs, metric
        else:
            return outputs

    @staticmethod
    def reshape_raw_y(env, y):
        raw_y = env.raw_y.values.reshape(-1, 1)
        if len(raw_y) > len(y):
            raw_y = raw_y[env.n_steps-1:]
        return raw_y