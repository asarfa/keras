from dataset_engineering.env import Env
import pandas as pd


def compute_envs(data: pd.DataFrame, features: list, train_dates, val_dates, data_test: pd.DataFrame = None,
                 fix_outliers: bool = True, add_transf: bool = True, allow_missing_timesteps: bool = False,
                 scale_target: bool = False, n_steps: int=100):
    train_env = Env(data, train_dates.start, train_dates.end, features,
                    fix_outliers=fix_outliers, add_transf=add_transf, allow_missing_timesteps=allow_missing_timesteps,
                    scale_target=scale_target, n_steps=n_steps)
    if scale_target:
        scaler_train_X, knn_imputer, scaler_outliers, scaler_train_y = train_env.scaler_X, train_env.knn_imputer, \
                                                                       train_env.scaler_outliers, train_env.scaler_y
    else:
        scaler_train_X, knn_imputer, scaler_outliers, scaler_train_y = train_env.scaler_X, train_env.knn_imputer, \
                                                                       train_env.scaler_outliers, None
    if data_test is not None:
        start_val, end_val = None, None
        data = data_test
    else:
        start_val, end_val = val_dates.start, val_dates.end
    eval_env = Env(data, start_val, end_val, features, scaler_train_X=scaler_train_X,
                  knn_imputer=knn_imputer, fix_outliers=fix_outliers, scaler_outliers=scaler_outliers,
                  add_transf=add_transf, allow_missing_timesteps=allow_missing_timesteps,
                  scale_target=scale_target, scaler_train_y=scaler_train_y, n_steps=n_steps)
    return train_env, eval_env
