import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from typing import List, Tuple, Union
import numpy as np
from pylab import plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None


class Env:
    def __init__(self,
                 data: pd.DataFrame,
                 start: int,
                 end: int,
                 features: list,
                 scaler_train_X: Union[MinMaxScaler, StandardScaler] = None,
                 knn_imputer: KNNImputer = KNNImputer(),
                 fix_outliers: bool = True,
                 scaler_outliers: StandardScaler = StandardScaler(),
                 add_transf: bool = True,
                 allow_missing_timesteps: bool = False,
                 scale_target: bool = False,
                 scaler_train_y: Union[MinMaxScaler, StandardScaler] = None,
                 scaler_outliers_y: StandardScaler = StandardScaler(),
                 n_steps: int = 100,
                 verbose: bool = False
                 ):
        self.data = data
        self.start = start
        self.end = end
        self.features = features
        self.knn_imputer = knn_imputer
        self.fix_outliers = fix_outliers
        self.scaler_outliers = scaler_outliers
        self.add_transf = add_transf
        self.allow_missing_timesteps = allow_missing_timesteps
        self.scale_target = scale_target
        self.scaler_outliers_y = scaler_outliers_y
        self.n_steps = n_steps
        self.target = 'spot_id_delta'
        self.verbose = verbose
        self.main(scaler_train_X, scaler_train_y)

    def index_data(self):
        # self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.iloc[self.start: self.end]
        if self.verbose:
            print('*' * 50)
            print(f'Data starting date = {self.data.index[0][:13]}h')
            print(f'Data ending date = {self.data.index[-1][:13]}h')
        self.X = self.data[self.features]
        if self.target in self.data.columns:
            self.y = self.data[self.target]
        else:
            self.y = None

    def info_data(self):
        if self.verbose:print('*' * 50)
        self.outliers = (np.abs((self.X - self.X.mean()) / self.X.std()) > 3)
        if self.y is not None: self.outliers_y = (np.abs((self.y - self.y.mean()) / self.y.std()) > 3)
        if self.verbose:print(f'Number of Outliers : \n {self.outliers.sum()}')
        self.missing_values = self.X.isnull().sum()
        if self.verbose:
            print('*' * 50)
            print(f'Missing Values: \n {self.missing_values}')
            print('*' * 50)
        ts_with_delta = pd.concat([pd.Series(self.data.index.to_list()),
                                   pd.Series(pd.to_datetime(self.data.index.to_list())).diff()], axis=1)
        self.missing_ts = ts_with_delta[1].value_counts().iloc[1:]
        if self.verbose:
            print(f'Missing TimeSteps : \n {self.missing_ts}')
            print('*' * 50)
            print(f'Correlation : \n {self.data.corr().iloc[:-1, -1]}')
            print('*' * 50)

    @staticmethod
    def init_scaler(scaler_train, type_scaler):
        if scaler_train is None:
            if type_scaler == "normalizer":
                return MinMaxScaler((-1, 1))
            else:
                return StandardScaler()
        else:
            return scaler_train

    def use_scaler(self, scaler_train, scaler, data):
        if scaler_train is None:
            if self.verbose:print(f'Fit_transform {type(scaler)}')
            return scaler.fit_transform(data)
        else:
            if self.verbose:print(f'Transform {type(scaler)}')
            return scaler.transform(data)

    def remove_outliers(self, scaler_train_X):
        cols = self.outliers.sum().index.to_list()
        zscore = pd.DataFrame(self.use_scaler(scaler_train_X, self.scaler_outliers, self.X),
                              index=self.X.index, columns=self.X.columns)
        for col in cols:
            if zscore.sum()[col] > 0:
                if self.verbose:print(f'Removing outliers for {col}')
                serie_zscore = zscore[col]
                serie_zscore_bool = np.abs(serie_zscore) > 3
                serie_w_outliers = self.X[col][~serie_zscore_bool]
                self.X.loc[:, col] = np.where(serie_zscore > 3, np.max(serie_w_outliers), self.X[col])
                self.X.loc[:, col] = np.where(serie_zscore < -3, np.min(serie_w_outliers), self.X[col])

    def remove_outliers_y(self, scaler_train_X):
        zscore = pd.DataFrame(self.use_scaler(scaler_train_X, self.scaler_outliers_y, self.y.to_frame()),
                              index=self.y.index, columns=[self.target]).T.squeeze()
        serie_zscore_bool = np.abs(zscore) > 3
        serie_w_outliers = self.y[~serie_zscore_bool]
        self.y = np.where(zscore > 3, np.max(serie_w_outliers), self.y)
        self.y = np.where(zscore < -3, np.min(serie_w_outliers), self.y)
        self.y = pd.Series(self.y, index=self.X.index)
        self.y.name = self.target

    def perform_inputation(self, scaler_train_X):
        self.X = pd.DataFrame(self.use_scaler(scaler_train_X, self.knn_imputer, self.X),
                              index=self.X.index, columns=self.X.columns)

    def fill_missing_ts(self):
        start_date = self.data.index[0][:13]
        end_date = self.data.index[-1][:13]

        tab = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq="h", tz="Europe/Paris"))
        tab.index.name = 'DELIVERY_START'
        # data.index = pd.to_datetime(data.index)
        tab.index = tab.index.astype('str')
        self.X = pd.merge(tab, self.X, on='DELIVERY_START', how='left').ffill().bfill()
        if self.y is not None:
            self.y = pd.merge(tab, self.y, on='DELIVERY_START', how='left').fillna(1)

    def form_data3d(self):
        X_array, y_array = self.X.values, self.y.values if self.y is not None else np.random.randn(len(self.X))
        batch_size = len(self.X) - self.n_steps + 1
        X_3d = np.zeros((batch_size, self.n_steps, len(self.features)))
        y_2d = y_array[self.n_steps-1:].reshape(-1, 1)
        for start_idx in range(batch_size):
            X_3d[start_idx] = X_array[start_idx: start_idx + self.n_steps]
        return X_3d, y_2d if self.y is not None else None

    def main(self, scaler_train_X, scaler_train_y):

        self.index_data()

        self.info_data()

        if self.y is not None: self.raw_y = self.y.copy()

        if self.fix_outliers and self.outliers.sum().sum() > 1:
            self.remove_outliers(scaler_train_X)
            if self.y is not None: self.remove_outliers_y(scaler_train_X)

        if self.missing_values.sum() > 1:
            self.perform_inputation(scaler_train_X)

        if self.add_transf:
            self.X = self.X ** 0.5

        self.scaler_X = self.init_scaler(scaler_train_X, "normalizer")
        self.X = pd.DataFrame(self.use_scaler(scaler_train_X, self.scaler_X, self.X),
                              index=self.X.index, columns=self.X.columns)

        if self.scale_target:
            self.scaler_y = self.init_scaler(scaler_train_y, "normalizer")
            if self.y is not None:
                self.y = pd.DataFrame(self.use_scaler(scaler_train_y, self.scaler_y, self.y.to_frame()),
                                   index=self.y.index, columns=[self.target]).T.squeeze()

        if self.allow_missing_timesteps and len(self.missing_ts) > 1:
            self.fill_missing_ts()

        self.X_tab = self.X.values.reshape(self.X.shape[0], 1, self.X.shape[1])
        self.y_tab = self.y.values.reshape(-1, 1) if self.y is not None else None
        self.X_ts, self.y_ts = self.form_data3d()

        if self.verbose:print("*************************Processed*************************")
