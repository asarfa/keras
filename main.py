import keras.src.metrics
import pandas as pd
from dataset_engineering.cross_val import Kfold, TSfold
from dataset_engineering.source import compute_envs
from neural_nets.baseline import Params
from neural_nets.model import Model
from finetuning.hypermodel import HyperModel
from finetuning.analyse import analyse_trials_opt, select_hp
import keras_tuner as kt
import tensorflow as tf
import numpy as np
import random
import os



if __name__ == '__main__':

    X = pd.read_csv('X_train.csv').drop(['predicted_spot_price'], axis=1)
    X.set_index('DELIVERY_START', inplace=True)
    features = X.columns.to_list()

    y = pd.read_csv('y_train.csv')
    y.set_index('DELIVERY_START', inplace=True)

    data = pd.concat([X, y], axis=1)

    X_test = pd.read_csv('X_test.csv').drop(['predicted_spot_price'], axis=1)
    X_test.set_index('DELIVERY_START', inplace=True)

    """
    cv = Kfold(data, gap)
    for i, walk in cv.get_walks(True):
        train_dates, val_dates = walk.train, walk.val
    """

    config_hp = ['lr', 'n_hidden', 'hidden_dim', 'dropout', 'l1reg', 'fix_outliers', 'scale_y']#, 'fix_outliers', 'scale_y', 'allow_missing_ts'

    params = Params(input_dim=len(features), criterion=tf.keras.losses.MeanSquaredError(), layer='DNN', n_steps=24*3, n_hidden=1,
                    epochs=20)#, gradient_clip=1)

    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    cv = TSfold(data, gap=params.n_steps)

    project_name = f'{cv.name}__{params.criterion.name}__{params.layer}__' + "__".join(config_hp)
    directory = "my_finetuning"
    max_trials = 20
    tuner = kt.RandomSearch(
        hypermodel=HyperModel(init_params=params, config=config_hp),
        objective=kt.Objective("val_mean_wacc", direction="max"),
        max_trials=max_trials,
        directory=directory,
        overwrite=False,
        project_name=project_name,
        seed=params.seed
    )
    tuner.search(data=data, features=features, cv=cv, params=params, project_name=project_name,
                 callbacks=[tf.keras.callbacks.TensorBoard(f'hp_logs/{project_name}', write_graph=False)])

    trials = analyse_trials_opt(max_trials, directory, project_name, sort_by='val_mean_wacc')

    cv = TSfold(data, gap=params.n_steps)
    for i, walk in cv.get_walks(False):
        train_dates, val_dates = walk.train, walk.val
    print('Setting train and val dates from last TS cross validation')

    for best_model_i in range(5):
        params_inf = select_hp(trials, config_hp, num=best_model_i, params=Params(**params.__dict__))
        fix_outliers = True if 'fix_outliers' in config_hp else False
        allow_missing_ts = True if 'allow_missing_ts' in config_hp else False
        scale_target = True if 'scale_y' in config_hp else False
        train_env, val_env = compute_envs(data.copy(), features, train_dates, val_dates, fix_outliers=fix_outliers,
                                          allow_missing_timesteps=allow_missing_ts, scale_target=scale_target)
        model = Model(train_env, val_env, params_inf, project_name, verbose=True, vis_board=True)
        model.fit()
        #metrics = model.evaluate(val_env)
        output_val, _ = model.infer_predict(val_env, save=True)
        _, test_env = compute_envs(data.copy(), features, train_dates, val_dates=None, data_test=X_test, fix_outliers=fix_outliers,
                                          allow_missing_timesteps=allow_missing_ts, scale_target=scale_target)
        #model.load_weights()
        model.infer_predict(test_env, save=True)

    dir = f"outputs/{project_name}/test"
    paths = os.listdir(dir)
    outputs = []
    missing_pred = pd.read_csv('y_random.csv').set_index('DELIVERY_START').iloc[:params.n_steps + 27]
    for path in paths:
        output = pd.read_csv(f'{dir}/{path}')
        output.set_index('DELIVERY_START', inplace=True)
        output.columns = ['spot_id_delta']
        output = pd.concat([missing_pred, output])
        output.to_csv(f'{dir}/{path}')
        outputs.append(output)
    output = pd.concat(outputs, axis=1).mean(axis=1).to_frame('spot_id_delta').to_csv(f'{dir}/bagging.csv')

    b

    for best_model_i in range(3):
        params = select_hp(trials, config_hp, num=best_model_i, params=Params(input_dim=len(features)))
        fix_outliers = True if 'fix_outliers' in config_hp else False
        allow_missing_ts = True if 'allow_missing_ts' in config_hp else False
        scale_target = True if 'scale_y' in config_hp else False
        train_env, test_env = compute_envs(data.copy(), features, train_dates, val_dates=None, data_test=X_test, fix_outliers=fix_outliers,
                                          allow_missing_timesteps=allow_missing_ts, scale_target=scale_target)
        model = Model(None, None, params, project_name, verbose=True)
        model.load_weights()
        model.infer_predict(test_env, save=True)


    """
    otp_dnn = pd.read_csv(r"agging.csv")
    otp_dnn.set_index('DELIVERY_START', inplace=True)
    otp_lstm = pd.read_csv(r"ix_outliers__scale_y\test\bagging.csv")
    otp_lstm.set_index('DELIVERY_START', inplace=True)
    pd.concat([otp_dnn.iloc[:len(X_test) - len(otp_lstm)], otp_lstm]).to_csv('first_submission')
    """







    print('End')