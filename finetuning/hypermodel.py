import keras_tuner as kt
from neural_nets.baseline import NN
from neural_nets.model import Model
from dataset_engineering.source import compute_envs
import numpy as np
import tensorflow as tf
import datetime
from keras.src.callbacks import TensorBoard, Callback
from keras_tuner.src.engine.tuner_utils import TunerCallback
from copy import deepcopy


class HyperModel(kt.HyperModel):
    def __init__(self, init_params, config):
        self.init_params = init_params
        self.config = config

    def build(self, hp):
        params = deepcopy(self.init_params)
        if 'hidden_dim' in self.config:
            params.hidden_dim = hp.Choice("hidden_dim", [8, 16])
        if 'n_hidden' in self.config:
            params.n_hidden = hp.Int("n_hidden", 1, 4)
        if 'dropout' in self.config:
            params.dropout = hp.Float("dropout", min_value=0.1, max_value=0.3, sampling="log")
        if 'lr' in self.config:
            params.lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        if 'l1reg' in self.config:
            params.l1reg = hp.Float("l1", min_value=0.0005, max_value=0.001, sampling="log")
        neural_net = NN(params=params)
        return neural_net.init()

    def fit(self, hp, model, data, features, cv, params, project_name, **kwargs):
        if 'batch_size' in self.config:
            params.batch_size = hp.Choice("batch_size", [16, 32, 64])
        fix_outliers = True if 'fix_outliers' in self.config else False
        allow_missing_ts = True if 'allow_missing_ts' in self.config else False
        scale_target = True if 'scale_y' in self.config else False
        nn_hp = model
        metrics = {'val_wacc': [], 'val_acc': []}
        callbacks = kwargs['callbacks'][:-1]
        count_step = 0
        for i, walk in cv.get_walks(True):
            train_dates, val_dates = walk.train, walk.val
            train_env, val_env = compute_envs(data, features, train_dates, val_dates, fix_outliers=fix_outliers,
                                              allow_missing_timesteps=allow_missing_ts, scale_target=scale_target)
            model = Model(train_env, val_env, params, nn_hp=nn_hp, project_name=project_name)
            history = model.fit(callbacks)
            print("********************infer********************")
            outputs, metric = model.infer_predict(val_env)
            metrics['val_wacc'].append(metric)
            #metrics['val_acc'].append(metrics_eval['acc'])
            callbacks = [TensorBoard(log_dir=callbacks[0].log_dir), Callback(), TunerCallback(trial=callbacks[2].trial,
                                                                                              tuner=callbacks[2].tuner)]
            #for TSCV, keep in memory last fold step and begin with it
            count_step += history.epoch[-1]
            callbacks[0]._train_step = count_step
            callbacks[0]._val_step = count_step
        val_mean_wacc = np.mean(metrics['val_wacc'])
        val_sharpe = val_mean_wacc / np.std(metrics['val_wacc'])
        #val_mean_acc = np.mean(metrics['val_acc'])
        writer = tf.summary.create_file_writer(f'{model.dirpath}/cross_val')
        with writer.as_default():
            dt = datetime.datetime.now()
            seq = int(dt.strftime("%Y%m%d%H%M")[8:])
            tf.summary.scalar('val_sharpe', val_sharpe, step=seq)
            tf.summary.scalar('val_mean_wacc', val_mean_wacc, step=seq)
            #tf.summary.scalar('val_mean_acc', val_mean_acc, step=seq)
        return {'val_sharpe': val_sharpe, 'val_mean_wacc': val_mean_wacc}#, 'val_mean_acc': val_mean_acc}









"""
TensorBoard
    def set_model(self, model):
        
        self._train_step = 0 if '_train_step' not in self.__dict__.keys() else self._train_step
        self._val_step = 0 if '_val_step' not in self.__dict__.keys() else self._val_step
        
    def on_epoch_end(self, epoch, logs=None):
        epoch += self._train_step
"""
