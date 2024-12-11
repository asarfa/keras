
from dataclasses import dataclass
import pandas as pd

@dataclass
class Set:
    idx: int
    start: int
    end: int


@dataclass
class Walk:
    train: Set
    val: Set


class Kfold:
    def __init__(self, data: pd.DataFrame, gap: int):
        self.size = len(data)//2
        self.gap = gap
        self.dates = data.index
        self.name = 'Kcv'

    def split(self):
        for i in range(3):
            if i == 0:
                start_train = 0
                end_train = self.size - 1
                start_val = self.size + self.gap
                end_val = -1
            elif i == 1:
                start_train = start_val
                end_train = end_val
                start_val = 0
                end_val = self.size - 1
            yield (start_train, end_train), (start_val, end_val)

    def get_walks(self, verbose: bool=True):
        idx = 0
        for train_index, val_index in self.split():
            idx += 1
            walk = Walk(train=Set(idx=idx, start=train_index[0], end=train_index[-1]),
                        val=Set(idx=idx, start=val_index[0], end=val_index[-1]))
            if verbose:
               print('*' * 20, f'{idx}th fold', '*' * 20)
               print(f'Training: {self.dates[walk.train.start][:13]}h to {self.dates[walk.train.end][:13]}h,'
                     f' size={len(self.dates[walk.train.start: walk.train.end])}')
               print(f'Validation: {self.dates[walk.val.start][:13]}h to {self.dates[walk.val.end][:13]}h,'
                     f' size={len(self.dates[walk.val.start: walk.val.end])}')
            yield idx, walk


class TSfold:
    def __init__(self, data: pd.DataFrame, gap: int, n_folds: int = 5):
        self.train_size = len(data)//2
        self.val_size = self.train_size//n_folds - gap//n_folds
        self.gap = gap
        self.n_folds = n_folds
        self.dates = data.index
        self.name = 'TScv'

    def split(self):
        i, start_train = 0, 0
        while True:
            end_train = self.train_size - 1 + self.val_size*i
            start_val = end_train + 1 + self.gap
            end_val = start_val + self.val_size
            yield (start_train, end_train), (start_val, end_val)
            i += 1
            if i == self.n_folds:
                break

    def get_walks(self, verbose: bool=True):
        idx = 0
        for train_index, val_index in self.split():
            idx += 1
            walk = Walk(train=Set(idx=idx, start=train_index[0], end=train_index[-1]),
                        val=Set(idx=idx, start=val_index[0], end=val_index[-1]))
            if verbose:
               print('*' * 20, f'{idx}th fold', '*' * 20)
               print(f'Training: {self.dates[walk.train.start][:13]}h to {self.dates[walk.train.end][:13]}h,'
                     f' size={len(self.dates[walk.train.start: walk.train.end])}')
               print(f'Validation: {self.dates[walk.val.start][:13]}h to {self.dates[walk.val.end][:13]}h,'
                     f' size={len(self.dates[walk.val.start: walk.val.end])}')
            yield idx, walk