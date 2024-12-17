import json
import math
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule, LightningDataModule, seed_everything
from attr import attrib, attrs
import shutil
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichModelSummary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
from copy import copy, deepcopy
from torch.distributions.normal import Normal
import functools
import gc
import time

pd.options.mode.chained_assignment = None 

PADDED_Y_VALUE = -2
PADDED_INDEX_VALUE = -2

"""
Architecture
"""


def first_arg_id(x, *y):
    return x


class FCModel(nn.Module):
    """
    This class represents a fully connected neural network model with given layer sizes and activation function.
    """

    def __init__(self, sizes, dropout, n_features):
        """
        :param sizes: list of layer sizes (excluding the input layer size which is given by n_features parameter)
        :param input_norm: flag indicating whether to perform layer normalization on the input
        :param activation: name of the PyTorch activation function, e.g. Sigmoid or Tanh
        :param dropout: dropout probability
        :param n_features: number of input features
        """
        super(FCModel, self).__init__()
        sizes.insert(0, n_features)
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.Identity()
        self.activation = nn.Identity()
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the FCModel.
        :param x: input of shape [batch_size, slate_length, self.layers[0].in_features]
        :return: output of shape [batch_size, slate_length, self.output_size]
        """
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return x


class LTRModel(nn.Module):
    """
    This class represents a full neural Learning to Rank model with a given encoder model.
    """

    def __init__(self, input_layer, encoder, output_layer):
        """
        :param input_layer: the input block (e.g. FCModel)
        :param encoder: the encoding block (e.g. transformer.Encoder)
        :param output_layer: the output block (e.g. OutputLayer)
        """
        super(LTRModel, self).__init__()
        self.input_layer = input_layer if input_layer else nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        self.output_layer = output_layer

    def prepare_for_output(self, x, mask, indices):
        """
        Forward pass through the input layer and encoder.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: encoder output of shape [batch_size, slate_length, encoder_output_dim]
        """
        return self.encoder(self.input_layer(x), mask, indices)

    def forward(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: model output of shape [batch_size, slate_length, output_dim]
        """
        return self.output_layer(self.prepare_for_output(x, mask, indices), mask)

    def score(self, x, mask, indices):
        """
        Forward pass through the whole LTRModel and item scoring.

        Used when evaluating listwise metrics in the training loop.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: scores of shape [batch_size, slate_length]
        """
        return self.output_layer.score(self.prepare_for_output(x, mask, indices))


class OutputLayer(nn.Module):
    """
    This class represents an output block reducing the output dimensionality to d_output.
    """

    def __init__(self, d_model, d_output):
        """
        :param d_model: dimensionality of the output layer input
        :param d_output: dimensionality of the output layer output
        :param output_activation: name of the PyTorch activation function used before scoring, e.g. Sigmoid or Tanh
        """
        super(OutputLayer, self).__init__()
        self.activation = lambda y_pred: (y_pred - torch.min(y_pred)) / (torch.max(y_pred) - torch.min(y_pred))
        self.d_output = d_output
        self.w_1 = nn.Linear(d_model, d_output)

    def forward(self, x, mask):
        """
        Forward pass through the OutputLayer.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length, self.d_output]
        """
        x = self.w_1(x).squeeze(dim=2)
        return torch.vstack(
            [torch.hstack([self.activation(x[i][mask[i] == False]), x[i][mask[i] == True]]) for i in range(x.size(0))])

    def score(self, x):
        """
        Forward pass through the OutputLayer and item scoring by summing the individual outputs if d_output > 1.
        :param x: input of shape [batch_size, slate_length, self.d_model]
        :return: output of shape [batch_size, slate_length]
        """
        if self.d_output > 1:
            return self.forward(x).sum(-1)
        else:
            return self.forward(x)


@attrs
class PositionalEncoding:
    strategy = attrib(type=str)
    max_indices = attrib(type=int)


def make_model(params):
    """
    Helper function for instantiating LTRModel.
    :param fc_model: FCModel used as input block
    :param transformer: transformer Encoder used as encoder block
    :param post_model: parameters dict for OutputModel output block (excluding d_model)
    :param n_features: number of input features
    :return: LTR model instance
    """
    fc_model = FCModel([params.hidden_dim], params.dropout, n_features=params.input_dim)  # type: ignore
    d_model = params.input_dim if not fc_model else fc_model.output_size
    transformer = make_transformer(n_features=d_model, N=params.n_hidden, d_ff=params.hidden_dim,
                                   h=params.attention_heads,
                                   positional_encoding=PositionalEncoding(strategy=params.pos_encoding,
                                                                          max_indices=params.slength))  # type: ignore
    model = LTRModel(fc_model, transformer, OutputLayer(d_model, d_output=params.output_dim))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def _make_positional_encoding(d_model: int, positional_encoding):
    """
    Helper function for instantiating positional encodings classes.
    :param d_model: dimensionality of the embeddings
    :param positional_encoding: config.PositionalEncoding object containing PE config
    :return: positional encoding object of given variant
    """
    if positional_encoding.strategy is None:
        return None


def clones(module, N):
    """
    Creation of N identical layers.
    :param module: module to clone
    :param N: number of copies
    :return: nn.ModuleList of module copies
    """
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Stack of Transformer encoder blocks with positional encoding.
    """

    def __init__(self, layer, N, position):
        """
        :param layer: single building block to clone
        :param N: number of copies
        :param position: positional encoding module
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.position = position

    def forward(self, x, mask, indices):
        """
        Forward pass through each block of the Transformer.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        if self.position:
            x = self.position(x, mask, indices)
        mask = mask.unsqueeze(-2)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """

    def __init__(self, features, eps=1e-6):
        """
        :param features: shape of normalized features
        :param eps: epsilon used for standard deviation
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # type: ignore
        self.b_2 = nn.Parameter(torch.zeros(features))  # type: ignore
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the layer normalization.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :return: normalized input of shape [batch_size, slate_length, output_dim]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    Please not that for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        """
        :param size: number of input/output features
        :param dropout: dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Forward pass through the sublayer connection module, applying the residual connection to any sublayer with the same size.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param sublayer: layer through which to pass the input prior to applying the sum
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        return x + self.dropout(
            sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder block made of self-attention and feed-forward layers with residual connections.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: input/output size of the encoder block
        :param self_attn: self-attention layer
        :param feed_forward: feed-forward layer
        :param dropout: dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Forward pass through the encoder block.
        :param x: input of shape [batch_size, slate_length, self.size]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, self.size]
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    Basic function for "Scaled Dot Product Attention" computation.
    :param query: query set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param key: key set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param value: value set of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    :param mask: padding mask of shape [batch_size, slate_length]
    :param dropout: dropout probability
    :return: attention scores of shape [batch_size, slate_size, n_attention_heads, attention_dim]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 1, float("-inf"))

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention block.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of attention heads
        :param d_model: input/output dimensionality
        :param dropout: dropout probability
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass through the multi-head attention block.
        :param query: query set of shape [batch_size, slate_size, self.d_model]
        :param key: key set of shape [batch_size, slate_size, self.d_model]
        :param value: value set of shape [batch_size, slate_size, self.d_model]
        :param mask: padding mask of shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        if mask is not None:
            # same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Feed-forward block.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: input/output dimensionality
        :param d_ff: hidden dimensionality
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feed-forward block.
        :param x: input of shape [batch_size, slate_size, self.d_model]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def make_transformer(N=6, d_ff=2048, h=8, dropout=0.1, n_features=136,
                     positional_encoding=None):
    """
    Helper function for instantiating Transformer-based Encoder.
    :param N: number of Transformer blocks
    :param d_ff: hidden dimensionality of the feed-forward layer in the Transformer block
    :param h: number of attention heads
    :param dropout: dropout probability
    :param n_features: number of input/output features of the feed-forward layer
    :param positional_encoding: config.PositionalEncoding object containing PE config
    :return: Transformer-based Encoder with given hyperparameters
    """
    c = deepcopy
    attn = MultiHeadedAttention(h, n_features, dropout)

    ff = PositionwiseFeedForward(n_features, d_ff, dropout)
    position = _make_positional_encoding(n_features, positional_encoding)
    return Encoder(EncoderLayer(n_features, c(attn), c(ff), dropout), N, position)


"""
Data Transformation
"""


class ToTensor(object):
    """
    Wrapper for ndarray->Tensor conversion.
    """

    def __call__(self, sample):
        """
        :param sample: tuple of three ndarrays
        :return: ndarrays converted to tensors
        """
        x, y, y_origin, indices = sample
        return torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32), torch.from_numpy(y_origin).type(torch.float32), torch.from_numpy(indices).type(torch.long)


class FixLength(object):
    """
    Wrapper for slate transformation to fix its length, either by zero padding or sampling.

    For a given slate, if its length is less than self.dim_given, x's and y's are padded with zeros to match that length.
    If its length is greater than self.dim_given, a random sample of items from that slate is taken to match the self.dim_given.
    """

    def __init__(self, dim_given):
        """
        :param dim_given: dimensionality of x after length fixing operation
        """
        assert isinstance(dim_given, int)
        self.dim_given = dim_given

    def __call__(self, sample):
        """
        :param sample: ndarrays tuple containing features, labels and original ranks of shapes
        [sample_length, features_dim], [sample_length] and [sample_length], respectively
        :return: ndarrays tuple containing features, labels and original ranks of shapes
            [self.dim_given, features_dim], [self.dim_given] and [self.dim_given], respectively
        """
        sample_size = len(sample[1])
        if sample_size < self.dim_given:  # when expected dimension is larger than number of observation in instance do the padding
            fixed_len_x, fixed_len_y, fixed_len_y_origin, indices = self._pad(sample, sample_size)
        elif sample_size == self.dim_given:
            fixed_len_x = sample[0]
            fixed_len_y = sample[1]
            fixed_len_y_origin = sample[2]
            indices = np.arange(sample_size)
        else:
            raise ValueError('No sample needed')
        return fixed_len_x, fixed_len_y, fixed_len_y_origin, indices

    def _sample(self, sample, sample_size):
        """
        Sampling from a slate longer than self.dim_given.
        :param sample: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_length, features_dim], [sample_length] and [sample_length], respectively
        :param sample_size: target slate length
        :return: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_size, features_dim], [sample_size] and [sample_size]
        """
        indices = np.random.choice(sample_size, self.dim_given, replace=False)
        fixed_len_y = sample[1][indices]
        if fixed_len_y.sum() == 0:
            if sample[1].sum() == 1:
                indices = np.concatenate(
                    [np.random.choice(indices, self.dim_given - 1, replace=False), [np.argmax(sample[1])]])
                fixed_len_y = sample[1][indices]
            elif sample[1].sum() > 0:
                return self._sample(sample, sample_size)
        fixed_len_x = sample[0][indices]
        return fixed_len_x, fixed_len_y, indices

    def _pad(self, sample, sample_size):
        """
        Zero padding a slate shorter than self.dim_given
        :param sample: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_length, features_dim], [sample_length] and [sample_length]
        :param sample_size: target slate length
        :return: ndarrays tuple containing features, labels and original ranks of shapes
            [sample_size, features_dim], [sample_size] and [sample_size]
        """
        fixed_len_x = np.pad(sample[0], ((0, self.dim_given - sample_size), (0, 0)), "constant")
        fixed_len_y = np.pad(sample[1], (0, self.dim_given - sample_size), "constant", constant_values=PADDED_Y_VALUE)
        fixed_len_y_origin = np.pad(sample[2], (0, self.dim_given - sample_size), "constant", constant_values=PADDED_Y_VALUE)
        indices = np.pad(np.arange(0, sample_size), (0, self.dim_given - sample_size), "constant",
                         constant_values=PADDED_INDEX_VALUE)
        return fixed_len_x, fixed_len_y, fixed_len_y_origin, indices


"""
Data Loading
"""


class LibSVMDataset(Dataset):
    """
    LibSVM Learning to Rank dataset.
    """

    def __init__(self, X, y, y_origin, query_ids, transform=None):
        """
        :param X: scipy sparse matrix containing features of the dataset of shape [dataset_size, features_dim]
        :param y: ndarray containing target labels of shape [dataset_size]
        :param query_ids: ndarray containing group (slate) membership of dataset items of shape [dataset_size, features_dim]
        :param transform: a callable defining an optional transformation called on the dataset
        """
        # X = X.toarray()

        _, indices, counts = np.unique(query_ids, return_index=True, return_counts=True)
        groups = np.cumsum(counts[np.argsort(indices)])

        self.X_by_qid = np.split(X, groups)[:-1]
        self.y_by_qid = np.split(y, groups)[:-1]
        self.y_origin_by_qid = np.split(y_origin, groups)[:-1]

        self.longest_query_length = max([len(a) for a in self.X_by_qid])

        print("loaded dataset with {} queries".format(len(self.X_by_qid)))
        print("longest query had {} documents".format(self.longest_query_length))

        # self.transform = transform

    @staticmethod
    def load_from_env(env):
        y = env.data[env.target].values if env.target is not None else np.zeros(len(env.data))
        y_origin = env.data['target'].values if env.target is not None else np.zeros(len(env.data))
        x = env.data[env.features + [env.meta, env.bench]].values if env.target is not None else env.data[env.features].values
        query_ids = env.data.index.values
        return x, y, y_origin, query_ids

    @classmethod
    def from_svm_file(cls, env, transform=None):
        """
        Instantiate a LibSVMDataset from a LibSVM file path.
        :param svm_file_path: LibSVM file path
        :param transform: a callable defining an optional transformation called on the dataset
        :return: LibSVMDataset instantiated from a given file and with an optional transformation defined
        """
        x, y, y_origin, query_ids = LibSVMDataset.load_from_env(env)  # load_svmlight_file(svm_file_path, query_id=True)
        print("loaded dataset and got x shape {}, y shape {} and query_ids shape {}".format(
            x.shape, y.shape, query_ids.shape))
        return cls(x, y, y_origin, query_ids, transform)

    def __len__(self):
        """
        :return: number of groups (slates) in the dataset
        """
        return len(self.X_by_qid)

    def __getitem__(self, idx):
        """
        :param idx: index of a group
        :return: ndarrays tuple containing features and labels of shapes [slate_length, features_dim] and [slate_length], respectively
        """
        X = self.X_by_qid[idx]
        y = self.y_by_qid[idx]
        y_origin = self.y_origin_by_qid[idx]

        sample = X, y, y_origin

        # if self.transform:
        #    sample = self.transform(sample)

        return sample

    @property
    def shape(self):
        """
        :return: shape of the dataset [batch_dim, document_dim, features_dim] where batch_dim is the number of groups
            (slates) and document_dim is the length of the longest group
        """
        batch_dim = len(self)
        document_dim = self.longest_query_length
        features_dim = self[0][0].shape[-1]
        return [batch_dim, document_dim, features_dim]


def load_libsvm_role(input_stream, role) -> LibSVMDataset:
    """
    Helper function loading a LibSVMDataset of a specific role.

    The file can be located either in the local filesystem or in GCS.
    :param input_path: LibSVM file directory
    :param role: dataset role (file name without an extension)
    :return: LibSVMDataset from file {input_path}/{role}.txt
    """
    ds = LibSVMDataset.from_svm_file(input_stream)
    print("{} DS shape: {}".format(role, ds.shape))
    return ds


def fix_length_to_longest_slate(ds: LibSVMDataset) -> Compose:
    """
    Helper function returning a transforms.Compose object performing length fixing and tensor conversion.

    Length fixing operation will fix every slate's length to maximum length present in the LibSVMDataset.
    :param ds: LibSVMDataset to transform
    :return: transforms.Compose object
    """
    print("Will pad to the longest slate: {}".format(ds.longest_query_length))
    return transforms.Compose([FixLength(int(ds.longest_query_length)), ToTensor()])


def load_libsvm_dataset_role(env, role: str) -> LibSVMDataset:
    """
    Helper function loading a single role LibSVMDataset
    :param role: the role of the dataset - specifies file name and padding behaviour
    :param input_path: directory containing the LibSVM files
    :param slate_length: target slate length of the training dataset
    :return: loaded LibSVMDataset
    """
    ds = load_libsvm_role(env, role)
    return ds


def dynamic_padding_collate(batch):
    X, y, y_origin = zip(*batch)
    max_len = max(list(map(lambda label: len(label), y)))
    transform = transforms.Compose([FixLength(max_len), ToTensor()])
    X, y, y_origin, indices = zip(*list(map(lambda sample: transform(sample), batch)))
    return torch.stack(X), torch.stack(y), torch.stack(y_origin), torch.stack(indices)


class MarketEnv:
    """
    This class allows to represent the financial market environment
    Prepares the target, features
    """

    def __init__(
            self,
            data: pd.DataFrame = None,
            features: list = None,
            target: str = None,
            start: int = None,
            end: int = None,
    ):
        self.data = data
        self.features = features
        self.target = target
        self.start = start
        self.end = end
        self.bench = 'numerai_bench_model'
        self.meta = 'numerai_meta_model'
        self.__set_args()

    def __init_data(self):
        """
        Indexing
        """
        self.data.index.name = 'date'
        self.data = self.data.loc[self.start: self.end]
        self.data.drop('era', axis=1, inplace=True)

    def __compute_data(self):
        self.n_features = len(self.features)
        self.len_per_day = self.data.groupby('date').size()
        self.dates = self.data.index.drop_duplicates().to_list()

    def __set_args(self):
        self.__init_data()
        self.__compute_data()

    def __check_params(self):
        assert type(self.start) is type(self.end), "Start and End data must be same type"
        if self.start is not None:
            assert self.start <= self.end, "Start of trading Nonsense"


class DataModule(LightningDataModule):
    def __init__(self,
                 train_env: MarketEnv = None,
                 val_env: MarketEnv = None,
                 test_env: MarketEnv = None,
                 batch_size: int = None,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['train_env', 'val_env', 'test_env'])
        self.train_env = train_env
        self.val_env = val_env
        self.test_env = test_env
        self.batch_size = batch_size
        self.n_features = self.train_env.n_features if self.train_env is not None else self.test_env.n_features

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_env = load_libsvm_dataset_role(self.train_env, 'train')
            if self.val_env is not None: self.val_env = load_libsvm_dataset_role(self.val_env, 'val')

        # Assign test dataset for use in dataloader(s)
        if stage == "predict" or stage == "test" or stage is None:
            self.test_env = load_libsvm_dataset_role(self.test_env, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_env, shuffle=False, batch_size=self.batch_size, num_workers=2,
                          collate_fn=dynamic_padding_collate)

    def val_dataloader(self):
        if self.val_env is not None:
            return DataLoader(self.val_env, shuffle=False, batch_size=1, num_workers=2)
        return []

    def predict_dataloader(self):
        return DataLoader(self.test_env, shuffle=False, batch_size=1, num_workers=2)


"""
Cross-Validation
"""


@dataclass
class Set:
    idx: int
    start: datetime
    end: datetime


@dataclass
class Walk:
    train: Set
    valid: Set
    test: Set


class WalkForward:
    def __init__(self,
                 data: pd.DataFrame,
                 min_train_size: int = 52 * 3, #252 * 2,
                 max_train_size: int = 52 * 3, # 252 * 2,
                 val_size: int = 13, #20 * 3,
                 gap: int = 5):
        self.min_train_size = min_train_size + gap - 1
        self.max_train_size = max_train_size + gap - 1
        self.val_size = val_size
        self.gap = gap + 1
        self.dates = np.unique(data.index)

    def split(self):
        i, start_train, breakk = 0, 0, False
        while True:
            start = self.val_size * i
            stop = self.min_train_size + self.val_size * (i + 1)
            mid = stop - self.val_size + 1
            if stop >= len(self.dates):
                stop = len(self.dates) - 1
                breakk = True
            if start > self.max_train_size - self.min_train_size: start_train += self.val_size
            yield (start_train, mid - self.gap), (mid, stop)
            i += 1
            if breakk: break

    def get_walks(self, verbose: bool = True):
        idx = 0
        for train_index, valid_index in self.split():
            idx += 1
            start_train = self.dates[train_index[0]]
            end_train = self.dates[train_index[-1]]
            start_valid = self.dates[valid_index[0]]
            end_valid = self.dates[valid_index[-1]]
            walk = Walk(train=Set(idx=idx, start=start_train, end=end_train),
                        valid=Set(idx=idx, start=start_valid, end=end_valid),
                        test=None)
            if verbose:
                print('*' * 20, f'{idx}th walking forward', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}, size={walk.train.end - walk.train.start + 1}')
                print(f'Validation: {walk.valid.start} to {walk.valid.end}, size={walk.valid.end - walk.valid.start + 1}')
            yield idx, walk


class WalkForwardFull:
    def __init__(self,
                 data: pd.DataFrame,
                 train_size: int = 52 * 3, #252 * 2,
                 val_size: int = 13, #20 * 3,
                 test_size: int =13,
                 gap: int = 5):
        self.train_size = train_size + gap - 1
        self.val_size = val_size
        self.test_size = test_size
        self.gap = gap + 1
        self.dates = np.unique(data.index)
        self.count = 0

    def split(self):
        i, start_train, breakk = 0, 0, False
        while True:
            stop_val = self.train_size + self.val_size * (i + 1)
            mid = stop_val - self.val_size + 1
            stop_test = stop_val + self.gap + self.test_size - 1
            if stop_test>= len(self.dates):
                stop_test = len(self.dates) - 1
                breakk = True
            if i == 1:
                start_train += self.train_size - self.gap + 2
            elif i > 1:
                start_train += self.val_size
            yield (start_train, mid - self.gap), (mid, stop_val), (stop_val+self.gap, stop_test)
            i += 1
            self.count = i
            if breakk: break

    def get_walks(self, verbose: bool = True):
        idx = 0
        for train_index, valid_index, test_index in self.split():
            idx += 1
            start_train = self.dates[train_index[0]]
            end_train = self.dates[train_index[-1]]
            start_valid = self.dates[valid_index[0]]
            end_valid = self.dates[valid_index[-1]]
            start_test = self.dates[test_index[0]]
            end_test = self.dates[test_index[-1]]
            walk = Walk(train=Set(idx=idx, start=start_train, end=end_train),
                        valid=Set(idx=idx, start=start_valid, end=end_valid),
                        test=Set(idx=idx, start=start_test, end=end_test))
            if verbose:
                print('*' * 20, f'{idx}th walking forward', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}, size={walk.train.end - walk.train.start + 1}')
                print(f'Validation: {walk.valid.start} to {walk.valid.end}, size={walk.valid.end - walk.valid.start + 1}')
                print(f'Test: {walk.test.start} to {walk.test.end}, size={walk.test.end - walk.test.start + 1}')
            yield idx, walk


class WalkForwardBlockFull:
    def __init__(self,
                 data: pd.DataFrame,
                 train_size: int = 52 * 3, #252 * 2,
                 val_size: int = 13, #20 * 3,
                 test_size: int =13,
                 gap: int = 5):
        self.train_size = train_size + gap - 1
        self.val_size = val_size
        self.test_size = test_size
        self.gap = gap + 1
        self.dates = np.unique(data.index)

    def split(self):
        i, start_train, breakk = 0, 0, False
        while True:
            start = self.val_size * i
            stop_val = self.train_size + self.val_size * (i + 1)
            mid = stop_val - self.val_size + 1
            stop_test = stop_val + self.gap + self.test_size - 1
            if stop_test>= len(self.dates):
                stop_test = len(self.dates) - 1
                breakk = True
            if start > 0: start_train += self.val_size
            yield (start_train, mid - self.gap), (mid, stop_val), (stop_val+self.gap, stop_test)
            i += 1
            if breakk: break

    def get_walks(self, verbose: bool = True):
        idx = 0
        for train_index, valid_index, test_index in self.split():
            idx += 1
            start_train = self.dates[train_index[0]]
            end_train = self.dates[train_index[-1]]
            start_valid = self.dates[valid_index[0]]
            end_valid = self.dates[valid_index[-1]]
            start_test = self.dates[test_index[0]]
            end_test = self.dates[test_index[-1]]
            walk = Walk(train=Set(idx=idx, start=start_train, end=end_train),
                        valid=Set(idx=idx, start=start_valid, end=end_valid),
                        test=Set(idx=idx, start=start_test, end=end_test))
            if verbose:
                print('*' * 20, f'{idx}th walking forward', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}, size={walk.train.end - walk.train.start + 1}')
                print(f'Validation: {walk.valid.start} to {walk.valid.end}, size={walk.valid.end - walk.valid.start + 1}')
                print(f'Test: {walk.test.start} to {walk.test.end}, size={walk.test.end - walk.test.start + 1}')
            yield idx, walk


def dump_object(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def del_file(file_path):
    open(file_path, 'w').close()
    os.remove(file_path)


def delete_files(path, arg: str = 'Numerai'):
    args_file = os.listdir(path)
    for file in args_file:
        if arg not in file:
            delete_filename = os.path.join(path, file)
            shutil.rmtree(delete_filename)


def compute_envs_fit(target: str, train_data: pd.DataFrame, walk, val_features: list = None, fold: int = None):
    name = f'financial_envs/{target}/cv_fold{fold}.pkl'
    try:
        envs = load_object(name)
        train_env, valid_env = envs
    except FileNotFoundError:
        train_env = MarketEnv(train_data, val_features, target, walk.train.start, walk.train.end)
        valid_env = MarketEnv(train_data, val_features, target, walk.valid.start, walk.valid.end)
    return train_env, valid_env


"""
Parameters
"""


@dataclass
class Params:
    """
    This class allows to define the parameters of the architecture and training of a neural nets
    """
    input_dim: int = 2
    hidden_dim: int = 32
    n_hidden: int = 1
    dropout: float = 0.1
    seed: int = 42
    output_dim: int = 1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    l2reg: float = None
    l1reg: float = None
    opt: str = 'adam'
    criterion = None
    attention_heads: int = 0
    gradient_clip_val: float = None
    n_envs: int = 2
    slength: int = 800
    pos_encoding: str = None
    cv: bool = False
    fold = None


"""
Utils fitting
"""


class FakeCallback(Callback):
    def on_init_start(self, trainer):
        pass

    def on_init_end(self, trainer):
        pass

    def on_train_end(self, trainer, pl_module):
        pass


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self, monitor: list = None):
        super().__init__()
        self.metrics = dict(map(lambda metric: (metric, []), monitor))

    def on_train_epoch_end(self, trainer, pl_module):
        metric = deepcopy(trainer.callback_metrics)
        for key in self.metrics.keys():
            metric_value = metric[key].cpu().detach().numpy() if key in metric.keys() else np.nan
            self.metrics[key].append(metric_value)


def metrics_best_epoch(trainer, metric: str = 'val_acc', ascending: bool = False):
    index = np.where(np.array([type(cb).__name__ for cb in trainer.callbacks]) == 'MetricsCallback')[0][0]
    metrics = pd.DataFrame.from_dict(trainer.callbacks[index].metrics, orient='index').T
    metrics = metrics.sort_values(by=metric, ascending=ascending)
    return metrics.iloc[0].astype(float)


def describe_metrics(metrics: list):
    des = pd.concat(metrics, axis=1).T.reset_index(drop=True).describe().T
    return des[['mean', 'std']]


def wandb_connect():
    wandb_api_key = "548bbaaf2cd49d5c8623d448e2614311a872bcb4"
    wandb_conx = wandb.login(key=wandb_api_key)
    print(f"Connected to Wandb online interface : {wandb_conx}")


def init_name_vparams(params, hp: list = None):
    sorted_params = {key: value for key, value in sorted(params.__dict__.items())}
    sorted_params = {k: v for k, v in sorted_params.items() if v is not None}
    if hp is None:
        name = "__".join([str(k) + '_' + str(round(v, 4)) for k, v in sorted_params.items() if
                          not k in ['input_dim', 'seed', 'output_dim', 'epochs', 'opt', 'fold', 'n_steps', 'criterion',
                                    'target']])
    else:
        name = "__".join([str(k) + '_' + str(round(v, 4)) for k, v in sorted_params.items() if k in hp])
    return name


def init_name_vconfig(config):
    sorted_config = {key: value for key, value in sorted(config.items())}
    name = "__".join(
        [str(k) + '_' + str(round(v, 4)) for k, v in sorted_config.items() if not k in ['params', 'verbose_shape']])
    return name


def init_envs(envs, params):
    if isinstance(envs, tuple):
        train_env, valid_env = envs
        params.n_envs = 3
    else:
        train_env = envs
        valid_env = None
        params.n_envs = 2
    return train_env, valid_env, params


def init_params(config: dict = {}, params: Params = None, index: str = None):
    if params is None: raise Exception('Should provide a Params')
    params = deepcopy(params)
    if index is None:
        for key, value in config.items():
            params.__dict__[key] = value
    else:
        for key in config.keys():
            params.__dict__[key] = config[key][index]
    return params


def re_init_params(params: Params = None, dm: DataModule = None):
    params.input_dim = dm.n_features
    return params


"""
Fitting 
"""


def init_args(model=None, config: dict = {}, envs: tuple = None, patience: int = 10, metric: str = 'val_acc',
              mode: str = 'max', params: Params = None, project_name: str = None, hp: list = None, wf_block: bool = True,
              version: str = ''):
    if config is not None:
        config = config._items
        for key in config.copy():
            if key not in hp:
                config.pop(key)
        name = init_name_vconfig(config)
        params = init_params(config, params)
    else:
        name = init_name_vparams(params, hp)
    learn_env, valid_env, params = init_envs(envs, params)
    dm = DataModule(learn_env, valid_env, None, batch_size=params.batch_size)
    params = re_init_params(params, dm)
    early_stop_callback = EarlyStopping(monitor=metric.replace('val', 'train') if valid_env is None else metric,
                                        min_delta=0.002,
                                        patience=patience,
                                        verbose=True,
                                        mode=mode,
                                        )
    dirpath = f'./lightning_logs{version}/{project_name}/{name}'
    fix = f"_train{params.fold}/" if params.fold is not None else '/'
    dirpath += fix
    #fake = True if params.cv is True else False
    checkpoint_callback = ModelCheckpoint(monitor=metric,
                                          save_top_k=2,
                                          save_last=True,
                                          verbose=True,
                                          dirpath=dirpath,
                                          filename='{epoch:02d}-{val_loss:.4f}-{' + metric + ':.4f}',
                                          mode=mode)# if not fake else FakeCallback()
    # max_epoch_trained=model.max_epoch_trained if hasattr(model,'max_epoch_trained') else None)
    metrics_callback = MetricsCallback(monitor=[metric.replace('val', 'train'), metric, 'val_corr', 'val_feat_exp'])
    if isinstance(model, type): model = model(params, learn_env.n_features)
    if not wf_block:
        if not model.load_ckpt(project_name, params, metric, mode, hp):
            pass
        else:
            model.delete_files(model.path)
    return model, dm, params, early_stop_callback, checkpoint_callback, metrics_callback, name


def init_trainer(model=None, config: dict = {}, envs: tuple = None, params: Params = None, wandb_b: bool = True,
                 patience: int = 10,
                 project_name: str = 'MC_skew', metric: str = 'val_acc', mode: str = 'max', tuning: bool = False,
                 hp: list = None, fast_dev_run: bool = False, wf_block : bool = True, version: str = ''):
    max_epoch_trained = model.max_epoch_trained if hasattr(model, 'max_epoch_trained') else params.epochs
    model, dm, params, early_stop_callback, checkpoint_callback, metrics_callback, name = init_args(model, config, envs,
                                                                                                    patience, metric,
                                                                                                    mode,
                                                                                                    params,
                                                                                                    project_name, hp,
                                                                                                    wf_block, version)
    if wandb_b:
        print('*' * 50)
        print(f'Model name is {name}')
        print('*' * 50)
        if not tuning: wandb.init(config=config, project=project_name, name=f'fold{params.fold}_{name}')
        if hasattr(model, 'model'):
            wandb.watch(model.model, criterion=model.criterion, log='all', log_graph=True)
        else:
            pass
        logger = WandbLogger(log_model='all')
    else:
        logger = None
    if max_epoch_trained < params.epochs:
        max_epochs = params.epochs - max_epoch_trained
    else:
        max_epochs = params.epochs
    trainer = Trainer(max_epochs=max_epochs,
                      callbacks=[early_stop_callback, checkpoint_callback, metrics_callback, LearningRateMonitor(),
                                 RichModelSummary(max_depth=1)],
                      logger=logger, gradient_clip_val=params.gradient_clip_val, fast_dev_run=fast_dev_run)
    return trainer, model, dm, params, name


def fit(model=None, config: dict = None, envs: tuple = None, params: Params = None, wandb_b: bool = True,
        patience: int = 10,
        project: str = '', metric: str = 'val_acc', mode: str = 'max', hp: list = [],
        fast_dev_run: bool = False, wf_block: bool = True, version: str= ''):
    trainer, model, dm, _, name = init_trainer(model, config, envs, params, wandb_b, patience, project, metric, mode,
                                               hp=hp, fast_dev_run=fast_dev_run, wf_block=wf_block, version=version)
    trainer.fit(model, dm)
    return trainer


"""
Wandb tuning
"""


def wandb_tuner(model_base=None, config=None, envs=None, patience=None, metric=None, mode=None, project=None,
                params=None, hp=None):
    with wandb.init(config=config, project=project) as run:
        config = wandb.config
        trainer, model, dm, params, name = init_trainer(model_base, config, envs, params, True, patience,
                                                        project, metric, mode, tuning=True, hp=hp)
        run.name = name
        trainer.fit(model, dm)
        best_metrics = metrics_best_epoch(trainer, metric, False if mode == 'max' else True)
        wandb.log({f"best_{metric}": best_metrics[metric]})


def wandb_cv_tuner(cv=None, config=None, patience=None, metric=None, mode=None, project=None, params=None,
                   model_base=None,
                   hp=None, target=None, data=None,
                   val_features=None):
    with wandb.init(config=config, project=project) as run:
        config = wandb.config
        all_metrics = []
        torch.cuda.empty_cache()
        gc.collect()
        verbose = True
        wf_block = False if isinstance(cv, WalkForwardFull) else True
        submissions = []
        for i, walk in cv.get_walks(verbose):
            train_env, valid_env = compute_envs_fit(target, data, walk, val_features, fold=i)
            print('*' * 50)
            envs = (train_env, valid_env)
            del train_env
            del valid_env
            params.fold = i if isinstance(cv, WalkForwardBlockFull) else None
            trainer, model, dm, params, name = init_trainer(model_base, config, envs, params, True, patience, project,
                                                       metric, mode, hp=hp, tuning=True, wf_block=wf_block)
            del envs
            run.name = name
            trainer.fit(model, dm)
            best_metrics = metrics_best_epoch(trainer, metric, False if mode == 'max' else True)

            test_env = MarketEnv(data, features, None, walk.test.start, walk.test.end)
            y = compute_pred(test_env, model_base, params, project_name, metric, mode, hp, delete=True if isinstance(cv, WalkForwardBlockFull) else False)

            submission = pd.Series(y, index=test_env.data['id']).to_frame('prediction')
            submissions.append(submission)
            test_env.data['era'] = test_env.data.index
            submission = submission.merge(test_env.data[['era', 'id', 'target', 'numerai_meta_model', 'numerai_bench_model']], on=['id'], how='left')
            per_era_corr = submission.groupby("era").apply(
                lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
            )
            per_era_mmc = submission.dropna().groupby("era").apply(
                lambda x: correlation_contribution(x[["prediction"]], x["numerai_meta_model"], x["target"])
            )
            per_era_bmc = submission.dropna().groupby("era").apply(
                lambda x: correlation_contribution(x[["prediction"]], x["numerai_bench_model"], x["target"])
            )
            corr_mean_test = per_era_corr.mean()
            mmc_mean_test = per_era_mmc.mean()
            bmc_mean_test = per_era_bmc.mean()
            if sum(bmc_mean_test.isna()) > 1:
                payout_test = corr_mean_test
            else:
                payout_test = corr_mean_test + 2*bmc_mean_test
            best_metrics['test_payout'] = payout_test[0]
            best_metrics['test_corr'] = corr_mean_test[0]
            #wandb.log({f"best_{metric}_{i}": best_metrics[metric]})
            #wandb.log({f"best_val_corr_{i}": best_metrics['val_corr']})
            #wandb.log({f"best_test_payout_{i}": best_metrics['test_payout']})
            wandb.log({f"best_test_corr_{i}": best_metrics['test_corr']})
            all_metrics.append(best_metrics)
            torch.cuda.empty_cache()
            gc.collect()
            # assert(metrics[metric]==trainer.callbacks[4].best_model_score.detach().numpy())
        summary = describe_metrics(all_metrics)
        scorer = summary['mean'].loc[f'test_payout'] / summary['std'].loc[f'test_payout']
        wandb.log({"k_fold_score": scorer})
        wandb.log({f'{metric}_mean': summary['mean'].loc[f'{metric}']})
        wandb.log({f'val_corr_mean': summary['mean'].loc['val_corr']})
        wandb.log({f'val_feat_exp_mean': summary['mean'].loc['val_feat_exp']})
        wandb.log({f'test_payout_mean': summary['mean'].loc[f'test_payout']})
        wandb.log({f'test_corr_mean': summary['mean'].loc['test_corr']})
        wandb.log({f'test_corr_sharpe': summary['mean'].loc['test_corr'] / summary['std'].loc['test_corr']})
        if not wf_block:
            path_save = f"predictions/{project_name}"
            os.makedirs(path_save, exist_ok=True)
            pd.concat(submissions).to_csv(f'{path_save}/{init_name_vparams(params, hp)}.csv')


def find_best_trials(path: str = None, params: Params = None, metric: str = "val_acc", reverse: bool = True):
    api = wandb.Api(api_key="548bbaaf2cd49d5c8623d448e2614311a872bcb4")
    sweep = api.sweep(path)
    runs = sorted(sweep.runs,
                  key=lambda run: run.summary.get(metric, 0), reverse=reverse)
    dic = {}
    k_fold_try = False
    for run in runs:
        config = json.loads(run.json_config)
        true_params = init_params(config, params, 'value')
        try:
            dic[run.id] = {metric: run.summary[metric],
                           'params': true_params}
        except KeyError:
            if metric == 'k_fold_score':
                results = np.array([run.summary[m] for m in run.summary.keys() if 'best_val' in m])
                if len(results) != 0:
                    dic[run.id] = {metric: np.mean(results) - np.std(results),
                                   'params': true_params}
                k_fold_try = True
            pass
    if k_fold_try: dic = dict(sorted(dic.items(), key=lambda item: item[1]['k_fold_score'], reverse=True))
    return dic


def save_best_models(project: str = "", sweep_id: str = '', params: Params = None,
                     metric: str = "val_acc",
                     reverse: bool = True, top_k: int = 5):
    path_wandb = 'numera/' + project + '/'
    path_save = f'./artifacts/{project}'
    if os.path.isdir(path_save) and len(os.listdir(path_save)) > 0:
        return None
    else:
        os.makedirs(path_save, exist_ok=True)
        trials = find_best_trials(path_wandb + sweep_id, params, metric, reverse)
        dirs = []
        if metric == 'k_fold_score':
            # trials = dict(map(lambda trial: (trial, trials[trial][metric]), trials.keys()))
            # trials = dict(sorted(trials.items(), key=lambda x: x[1], reverse=reverse))
            for i, trial in enumerate(trials.keys()):
                value_metric = trials[trial][metric]
                params = trials[trial]['params']
                print(f'Best model n°{i}: {metric}={round(value_metric, 3)}\n'
                      f'Configuration is: {params}')
                print('*' * 50)
                new_dir = path_save + f'/model{i}'
                os.makedirs(new_dir, exist_ok=True)
                with open(new_dir + '/params.pkl', 'wb') as file:
                    pickle.dump(params, file)
                dirs.append(new_dir)
                if i >= top_k - 1:
                    break
        else:
            run = wandb.init()
            sub_trials = {}
            for trial, values in trials.items():
                artifact = run.use_artifact(path_wandb + 'model-' + trial + ':best', type='model')
                if int(artifact.version.split('v')[1]) >= 2:  # ensure that the training had at least 2 epochs
                    sub_trials[trial] = artifact.metadata['score']
            sub_trials = dict(sorted(sub_trials.items(), key=lambda x: x[1], reverse=reverse))
            for i, trial in enumerate(sub_trials.keys()):
                artifact = run.use_artifact(path_wandb + 'model-' + trial + ':best', type='model')
                value_metric = trials[trial][metric]
                params = trials[trial]['params']
                if artifact.metadata['ModelCheckpoint']['monitor'] == metric:
                    value_metric_best_epoch = sub_trials[trial]
                    if value_metric_best_epoch != value_metric:
                        print(
                            f'Best model n°{i}: at best epoch {metric}={round(value_metric_best_epoch, 3)} vs at last epoch {metric}={round(value_metric, 3)}\n'
                            f'Configuration is: {params}')
                        print('*' * 50)
                artifact_dir = artifact.download()
                new_dir = path_save + f'/model{i}'
                os.rename(artifact_dir, new_dir)
                with open(new_dir + '/params.pkl', 'wb') as file:
                    pickle.dump(params, file)
                dirs.append(new_dir)
                if i >= top_k - 1:
                    break
        print(f'Models checkpoint save at: {dirs}')
        return dirs


def load_model(num: int = 0, project: str = None, version: str = ''):
    path = f'./artifacts{version}/{project}/model{num}'
    with open(path + '/params.pkl', 'rb') as file:
        params = pickle.load(file)
    return path + '/model.ckpt', params


def params_instanciation(project_name: str = None, params: Params = None, model_i: int = 0, version: str = ''):
    loss = params.criterion
    path, params = load_model(model_i, project_name, version)
    params.criterion = loss
    print(
        f'Best model n°{model_i} from tuning: \n'
        f'Configuration is: {params}')
    print('*' * 50)
    return params


"""
Metric
"""


def NumeraiCorr(y_pred: torch.tensor, y_true: torch.tensor):
    return torch.tensor(numerai_corr(pd.DataFrame(y_pred.detach().cpu().numpy(), columns=['prediction']),
                                             pd.Series(y_true.detach().cpu().numpy().squeeze(), name='target')).values[
                            0])


def NumeraiFNC(y_pred: torch.tensor, y_true: torch.tensor, features: torch.tensor):
    return torch.tensor(feature_neutral_corr(pd.DataFrame(y_pred.detach().cpu().numpy(), columns=['prediction']),
                                             pd.DataFrame(features.detach().cpu().numpy()),
                                             pd.Series(y_true.detach().cpu().numpy().squeeze(), name='target')).values[
                            0])


def NumeraiMMC(y_pred, y_true, meta_model):
    try:
        mmc = torch.tensor(correlation_contribution(pd.DataFrame(y_pred.detach().cpu().numpy(), columns=['prediction']),
                                                    pd.Series(meta_model.detach().cpu().numpy().squeeze(),
                                                              name='meta_model'),
                                                    pd.Series(y_true.detach().cpu().numpy().squeeze(),
                                                              name='target')).values[0])
    except:
        mmc = torch.tensor(0.0)
    return mmc


def NumeraiBMC(y_pred, y_true, bench_model):
    try:
        bmc = torch.tensor(correlation_contribution(pd.DataFrame(y_pred.detach().cpu().numpy(), columns=['prediction']),
                                                    pd.Series(bench_model.detach().cpu().numpy().squeeze(),
                                                              name='bench_model'),
                                                    pd.Series(y_true.detach().cpu().numpy().squeeze(),
                                                              name='target')).values[0])
    except:
        bmc = torch.tensor(0.0)
    return bmc


"""
Loss
"""


def pearson_correlation(y_pred, y_true, dim=1):
    mask = y_true == PADDED_Y_VALUE
    y_true[mask] = 0.0

    # Calculate the mean of the target and prediction

    mean_true = torch.mean(y_true, dim=dim)
    mean_pred = torch.mean(y_pred, dim=dim)
    # Calculate the centered values
    centered_true = y_true - mean_true.unsqueeze(1).expand(-1, y_true.size(1))
    centered_pred = y_pred - mean_pred.unsqueeze(1).expand(-1, y_true.size(1))
    # Calculate the variance of th target and prediction
    true_sqsum = torch.sum(torch.square(centered_true), dim=dim)
    pred_sqsum = torch.sum(torch.square(centered_pred), dim=dim)
    # Calculate the Pearson correlation coefficient
    cov = torch.sum(centered_pred * centered_true, dim=dim)
    return cov / torch.sqrt(pred_sqsum * true_sqsum)


def spearman_correlation_loss(y_pred, y_true, dim=1):
    """
    Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
   while trying to have the same mean and variance
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == PADDED_Y_VALUE
    y_true[mask] = 0.0
    y_pred[mask] = 0.0

    # Calculate the mean of the target and prediction

    mean_true = torch.mean(y_true, dim=dim)
    mean_pred = torch.mean(y_pred, dim=dim)
    # Calculate the centered values
    centered_true = y_true - mean_true.unsqueeze(1).expand(-1, y_true.size(1))
    centered_pred = y_pred - mean_pred.unsqueeze(1).expand(-1, y_true.size(1))
    # Calculate the variance of th target and prediction
    true_sqsum = torch.sum(torch.square(centered_true), dim=dim)
    pred_sqsum = torch.sum(torch.square(centered_pred), dim=dim)
    # Calculate the Pearson correlation coefficient
    cov = torch.sum(centered_pred * centered_true, dim=dim)
    corr = cov / torch.sqrt(pred_sqsum * true_sqsum)
    n = torch.tensor(y_pred.shape[-2], dtype=y_pred.dtype)
    sqdif = torch.sum((y_pred - y_true) ** 2, dim=dim) / n / torch.sqrt(true_sqsum / n)
    spearman_corr = torch.tensor(1.0, dtype=y_pred.dtype) - corr + (0.01 * sqdif)
    # Add a small epsilon to avoid division by zero return pearson_corr

    return torch.mean(spearman_corr, dtype=torch.float32)


def concordance_correlation_loss(y_pred, y_true, dim=1):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == PADDED_Y_VALUE
    y_true[mask] = 0.0
    y_pred[mask] = 0.0
    # Calculate the mean of the target and prediction

    mean_true = torch.mean(y_true, dim=dim)
    mean_pred = torch.mean(y_pred, dim=dim)
    # Calculate the centered values
    centered_true = y_true - mean_true.unsqueeze(1).expand(-1, y_true.size(1))
    centered_pred = y_pred - mean_pred.unsqueeze(1).expand(-1, y_true.size(1))
    # Calculate the variance of th target and prediction
    true_sqsum = torch.sum(torch.square(centered_true), dim=dim)
    pred_sqsum = torch.sum(torch.square(centered_pred), dim=dim)
    # Calculate the Pearson correlation coefficient
    cov = torch.sum(centered_pred * centered_true, dim=dim)
    corr = cov / torch.sqrt(pred_sqsum * true_sqsum)  ##=-pearson_corr

    concor_corr = (2 * corr * torch.sqrt(pred_sqsum * true_sqsum)) / (
            pred_sqsum + true_sqsum + (mean_pred - mean_true) ** 2)

    return - torch.mean(concor_corr, dtype=torch.float32)


def mse_loss(y_pred, y_true, dim=1):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == PADDED_Y_VALUE
    y_true[mask] = 0.0
    y_pred[mask] = 0.0
    mse = nn.MSELoss()
    return mse(y_pred, y_true)


"""
Linear achitecture
"""


class Layers:
    def __init__(self, params: Params):
        self.params = params
        torch.manual_seed(self.params.seed)

    def dropout(self):
        return nn.Dropout(p=self.params.dropout)

    def linear(self):
        return nn.Linear(in_features=self.params.input_dim, out_features=self.params.hidden_dim)

    def relu(self):
        return nn.ReLU()

    def set_linear(self, stack_layers: list):
        stack_layers.append(self.linear())
        stack_layers.append(self.relu())
        if self.params.dropout != 0: stack_layers.append(self.dropout())
        return stack_layers

    def set(self, _type: str = None, stack_layers: list = None):
        if _type == 'DNN':
            stack_layers = self.set_linear(stack_layers)
        return stack_layers

    def get(self, _type: str):
        stack_layers = []
        stack_layers = self.set(_type, stack_layers)
        true_input_dim = self.params.input_dim
        for i in range(self.params.n_hidden):
            self.params.input_dim = self.params.hidden_dim
            stack_layers = self.set(_type, stack_layers)
        if hasattr(self.params, 'augmented_fcl') and self.params.augmented_fcl == 1:
            self.params.input_dim = self.params.hidden_dim
            stack_layers = self.set('DNN', stack_layers)
        self.params.input_dim = true_input_dim
        if self.params.dropout != 0:
            return stack_layers[:-1]  # Dropout layer on the outputs of each layer except the last layer
        else:
            return stack_layers


class Linear(nn.Module):
    def __init__(self, params: Params):
        super(Linear, self).__init__()
        self.params = params
        self.stack_layers = Layers(params).get('DNN')
        self.linear = nn.Linear(self.params.hidden_dim, self.params.output_dim)
        self.compute_set()

    def compute_set(self):
        for i, layer in enumerate(self.stack_layers):
            name = f'Layer_{str(i + 1).zfill(3)}'
            setattr(self, name, layer)

    def forward(self, x):
        for i, layer in enumerate(self.stack_layers):
            x = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
        x = self.linear(x)
        return x.squeeze(2)


"""
Model class
"""


class Base:
    def __init__(self):
        self.device = None
        self.model = None

    def load_ckpt(self, project_name: str = None, params: Params = None, metric: str = 'val_acc',
                  mode: str = 'max', hp: list = None, version: str = ''):
        if params is not None:
            ascending = False if mode == 'max' else True
            is_saved = self._load_args(None, params, project_name, metric, ascending, hp, version)
        return is_saved

    def _load_args(self, path: str = None, params: Params = None, project_name: str = None, metric: str = 'val_acc',
                   ascending: bool = False, hp: list = None, version: str = ''):
        path_base = f'lightning_logs{version}/{project_name}/{init_name_vparams(params, hp)}'
        fix = f"_train{params.fold}/" if params.fold is not None else '/'
        path_base += fix
        check_exist = self._select_best_model(path_base, metric, ascending)
        if check_exist is False:
            return False
        else:
            self.path = path_base + check_exist
            if self.device.type == 'cpu':
                checkpoint = torch.load(self.path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(self.path)
            try:
                self.model.load_state_dict(checkpoint['state_dict'])
            except:
                self.load_state_dict(checkpoint['state_dict'])
            return True

    def _select_best_model(self, path: str = None, metric: str = 'val_acc', ascending: bool = False):
        if os.path.exists(path) and not '.ckpt' in path:
            self.delete_files(path, f'-v.ckpt')
            for i in range(1, 5):
                self.delete_files(path, f'-v{i}.ckpt')
            paths = os.listdir(path)
            if 'last.ckpt' in paths:
                paths.remove('last.ckpt')
                best_path = None
                best_metric = 1e10 if ascending else -1e10
                for path in paths:
                    epoch = int(path.split('epoch=')[1].split('-')[0])
                    metric_val = float(path.split(f'{metric}=')[1].split('ckpt')[0][:-1])
                    if epoch >= 0:
                        if not ascending:
                            if metric_val > best_metric:
                                best_metric = metric_val
                                best_path = path
                                self.max_epoch_trained = epoch
                        else:
                            if metric_val < best_metric:
                                best_metric = metric_val
                                best_path = path
                                self.max_epoch_trained = epoch
                print(f'Loading best model whose {metric}={best_metric}, {best_path}')
                return best_path if not best_path is None else 'last.ckpt'
            else:
                return False
        else:
            return False

    def delete_files(self, path, arg: str = ''):
        if 'ckpt' in path:
            delete_filename = '/'.join(path.split('/')[:-1])
            shutil.rmtree(delete_filename)
            print(f'Deleting {delete_filename}')
        else:
            args_file = os.listdir(path)
            for file in args_file:
                if arg in file:
                    delete_filename = os.path.join(path, file)
                    del_file(delete_filename)
                    print(f'Deleting {delete_filename}')


class BaseModel(LightningModule, Base):
    def __init__(self, params: Params = None,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['params'])
        self.lr = params.lr
        self.l2reg = params.l2reg
        self.l1reg = params.l1reg
        self.criterion = params.criterion
        self.monitor = 'val_loss' if params.n_envs == 3 else 'train_loss'
        seed_everything(params.seed)

    def compute_metrics(self, pred, true, X, type_env, loss):
        pred = self.compute_prediction(pred)
        mask = true != PADDED_Y_VALUE
        metrics = {}
        for name_metric, metric in self.metrics.items():
            if name_metric == 'corr':
                metrics[f'{type_env}_{name_metric}'] = torch.mean(torch.tensor([metric(pred[i][mask[i]],
                                                                                       true[i][mask[i]])
                                                                                for i in range(pred.size(0))]))
            if name_metric == 'fnc' and type_env != 'train':
                metrics[f'{type_env}_{name_metric}'] = torch.mean(torch.tensor([metric(pred[i][mask[i]],
                                                                                       true[i][mask[i]],
                                                                                       X[i][:, :-2][mask[i]])
                                                                                for i in range(pred.size(0))]))
            if name_metric == 'bmc':
                metrics[f'{type_env}_{name_metric}'] = torch.mean(torch.tensor([metric(pred[i][mask[i]],
                                                                                       true[i][mask[i]],
                                                                                       X[i][:, -1][mask[i]])
                                                                                for i in range(pred.size(0))]))
            if name_metric == 'mmc':
                metrics[f'{type_env}_{name_metric}'] = torch.mean(torch.tensor([metric(pred[i][mask[i]],
                                                                                       true[i][mask[i]],
                                                                                       X[i][:, -2][mask[i]])
                                                                                for i in range(pred.size(0))]))
            if name_metric == 'payout':
                metrics[f'{type_env}_{name_metric}'] = metrics[f'{type_env}_corr'] + 2*metrics[f'{type_env}_bmc']
            if name_metric == 'feat_exp':
                metrics[f'{type_env}_{name_metric}'] = torch.mean(torch.tensor([self.compute_feature_exposure(
                    X[i][:, :-2][mask[i]].unsqueeze(0), pred[i][mask[i]].unsqueeze(0)) for i in range(pred.size(0))]))
        metrics[f'{type_env}_loss'] = loss
        return metrics

    def compute_l1(self):
        l1_regularization = torch.tensor(0., device=self.device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_regularization += torch.norm(param, p=1)
        return self.l1reg * l1_regularization

    def compute_feature_exposure(self, features, pred):
        exposures = list(map(lambda f: pearson_correlation(pred, features[:, :, f]),
                             range(features.shape[-1])))
        tensor = torch.tensor(exposures)
        return torch.max(torch.abs(tensor[~tensor.isnan()]))

    def step(self, batch, batch_idx, type_env):
        if type_env == 'train':
            X, y, y_origin, indices = batch
        else:
            X, y, y_origin = batch
            X = X.type(torch.float32)
            indices = torch.arange(y.shape[1], dtype=torch.long).unsqueeze(0)
        mask = (y == PADDED_Y_VALUE)
        outputs = self.model.forward(X[:, :, :-2] / 4, mask, indices) if type_env != 'test' else self.model.forward(X/4, mask, indices)
        loss = self.criterion(outputs, y)
        if self.l1reg is not None: loss += self.compute_l1()
        # loss += self.compute_feature_exposure(X[:, :, :-2], outputs)
        if type_env != 'test':
            metrics = self.compute_metrics(outputs, y_origin, X, type_env, loss)
            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        return outputs, loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        _, loss = self.step(batch, batch_idx, type_env='train')
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        _, loss = self.step(batch, batch_idx, type_env='val')
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs, loss = self.step(batch, batch_idx, type_env='test')
        return self.compute_prediction(outputs).cpu().numpy()

    def test_step(self, batch, batch_idx):
        return self.predict_step(batch, batch_idx)

    def init_optimizer(self, opt: str):
        if opt == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif opt == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.l2reg)
        elif opt == 'ranger':
            pass
            # return Ranger21(self.parameters(), lr=self.lr, weight_decay=self.l2reg)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scheduler_config = {
            "scheduler": ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=3,
            ),
            "monitor": self.monitor,  # Default: val_loss
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_config}

    def get_test_metrics(self):
        for key in self.test_metrics.keys():
            self.test_metrics[key] = np.mean(self.test_metrics[key])
        return self.test_metrics


class LitTransformer(BaseModel):
    def __init__(self,
                 params: Params = None,
                 verbose_shape=None,
                 ):
        super().__init__(params=params)
        self.model = make_model(params)
        self.metrics = {
            'corr': NumeraiCorr,
            # 'fnc': NumeraiFNC,
            'bmc': NumeraiBMC,
            'mmc': NumeraiMMC,
            'payout': None,
            'feat_exp': None
        }
        self.test_metrics = {'test_' + key: [] for key in self.metrics.keys()}
        self.optimizer = self.init_optimizer(params.opt)

    def compute_prediction(self, pred):
        return pred


class LitLinear(BaseModel):
    def __init__(self,
                 params: Params = None,
                 verbose_shape=None,
                 ):
        super().__init__(params=params)
        self.model = Linear(params)
        self.metrics = {
            'corr': NumeraiCorr,
            # 'fnc': NumeraiFNC,
            'bmc': NumeraiBMC,
            'mmc': NumeraiMMC,
            'payout': None,
            'feat_exp': None
        }
        self.test_metrics = {'test_' + key: [] for key in self.metrics.keys()}
        self.optimizer = self.init_optimizer(params.opt)

    def compute_prediction(self, pred):
        return pred


def compute_fit(envs=None, model_base=None, project_name: str = None, metric: str = None, mode: str = None,
                params: Params = None,
                name_save: str = None, patience_es: int = 10, wandb_b: bool = True, hp: list = [],
                fast_dev_run: bool = False, wf_block: bool = True, version: str = ''):
    params.n_envs = 3
    fit_env, test_env = envs, None
    params.input_dim = fit_env.n_features if isinstance(fit_env, MarketEnv) else fit_env[0].n_features
    model = model_base(params)
    if not wf_block:
        if not model.load_ckpt(project_name, params, metric, mode, hp, version):
            pass
        else:
            model.delete_files(model.path)
    if not model.load_ckpt(project_name, params, metric, mode, hp, version):
        fit(envs=fit_env, model=model, patience=patience_es, metric=metric, mode=mode, project=project_name,
            params=params, wandb_b=wandb_b, hp=hp, fast_dev_run=fast_dev_run, wf_block=wf_block, version=version)


def compute_pred(test_env: MarketEnv = None, model_base=None, params: Params = None, project_name: str = None,
                 metric: str = None, mode: str = None, hp: list = None, delete: bool = False):
    start = time.time()
    dm = DataModule(None, None, test_env)
    params = re_init_params(params, dm)
    model = model_base(params)
    model.load_ckpt(project_name, params, metric, mode, hp)
    trainer = Trainer()
    y = trainer.predict(model, dm)
    y = np.hstack(y).squeeze()
    if delete: model.delete_files(model.path)
    return y


def compute_cv_tuning(sweep_config: dict, train_data: pd.DataFrame, model_base, params: Params, val_features: list,
                      target: str, metric: str, mode: str, project_name: str,
                      patience_es: int = 10, n_trials: int = 20, top_k: int = 5, cv=None):
    params.cv = True
    wandb_connect()
    hp = list(sweep_config['parameters'].keys())
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id,
                functools.partial(wandb_cv_tuner, model_base=model_base, data=train_data, val_features=val_features,
                                  target=target, cv=cv,
                                  patience=patience_es, metric=metric, mode=mode,
                                  project=project_name, params=params, hp=hp),
                count=n_trials)
    save_best_models(project_name, sweep_id, params, 'k_fold_score', top_k=top_k)
    return sweep_id, hp

def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=None)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + 0.5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1 + 0.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class']
    ax.set(yticks=np.arange(n_splits + 1) + 0.5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 1.2, -.1], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


def plot_cv(cv, X, y):
    this_cv = cv
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_cv_indices(this_cv, X, y, ax, cv.n_splits)

    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Validation set', 'Training set'], loc=(1.02, .8))
    plt.tight_layout()
    fig.subplots_adjust(right=.7)
    plt.show()


import logging
from numerapi import NumerAPI

TRAINED_MODEL_PREFIX = './trained_model'
MODEL_ID = 'Ensemble_LTR_16'

napi = NumerAPI()

model_name = TRAINED_MODEL_PREFIX
if MODEL_ID:
    model_name += f"_{MODEL_ID}"


@dataclass
class LoopParameters:
    train_i: int = 15
    live_round: int = 0

    def dump(self, name_save):
        dump_object(self, name_save)

    @classmethod
    def load(cls, name_save):
        try:
            loaded_instance = load_object(name_save)
            return loaded_instance
        except:
            return cls()


def download_data(version: str = 'v4.3'):
    print('Downloading dataset files...')

    napi.download_dataset(f"{version}/train_int8.parquet")
    napi.download_dataset(f"{version}/validation_int8.parquet")
    napi.download_dataset(f"{version}/features.json")
    napi.download_dataset(f"{version}/live_int8.parquet")
    napi.download_dataset("v4.3/meta_model.parquet")
    napi.download_dataset("v4.3/train_benchmark_models.parquet")
    napi.download_dataset("v4.3/validation_benchmark_models.parquet")

    feature_metadata = json.load(open(f"{version}/features.json"))
    features = feature_metadata["feature_sets"]["medium"]
    train = pd.read_parquet(f"{version}/train_int8.parquet", columns=["era"] + features + ["target"])
    val = pd.read_parquet(f"{version}/validation_int8.parquet", columns=["era"] + features + ["target"])
    data = pd.concat([train, val[~val['target'].isna()]])
    data['id'] = data.index
    data.index.name = 'days'

    meta_model = pd.read_parquet(f"v4.3/meta_model.parquet", columns=['era', 'numerai_meta_model'])

    train_bench = pd.read_parquet("v4.3/train_benchmark_models.parquet")
    val_bench = pd.read_parquet("v4.3/validation_benchmark_models.parquet")
    bench_name = 'v42_rain_ensemble'  # 'v43_lgbm_teager20'
    bench = pd.concat([train_bench, val_bench[~val_bench[bench_name].isna()]])[['era', bench_name]]
    bench.columns = ['era', 'numerai_bench_model']

    data = data.merge(meta_model, on=['id', 'era'], how='left')
    data = data.merge(bench, on=['id', 'era'], how='left')
    data.index = data.era.astype(int)

    data = data.loc[888 - 155 - 5:]

    del train
    del val

    live_data = pd.read_parquet(f"{version}/live_int8.parquet", columns=["era"] + features + ["target"])
    live_data['id'] = live_data.index
    live_data.index.name = 'days'
    live_data.index = live_data.era.replace('X', 0)
    return data, live_data, features


def train(project_name, params, model_i, target, data, walk, features, model_base, metric, mode, patience_es, hp, wf_block, version):
    print('Training model...')
    params = params_instanciation(project_name, params, model_i, version)
    train_env, valid_env = compute_envs_fit(target, data, walk, features, fold=None)
    compute_fit((train_env, valid_env), model_base, project_name, metric, mode, params,
                name_save=None, patience_es=patience_es, wandb_b=False, hp=hp,
                fast_dev_run=False, wf_block=wf_block, version=version)

    gc.collect()


def predict(model_i, live_data, features, model_base, params, project_name, metric, mode, hp):
    print('Making a prediction...')
    params = params_instanciation(project_name, params, model_i)
    test_env = MarketEnv(live_data, features, None, None, None)
    y = compute_pred(test_env, model_base, params, project_name, metric, mode, hp)
    submission = pd.Series(y, index=test_env.data['id']).to_frame('prediction')

    live_data.loc[:, f"prediction"] = submission.values

    #live_data["prediction"] = live_data[f"preds_{model_name}"].rank(pct=True)
    logging.info(f'Live predictions and ranked')

    gc.collect()

    return live_data


def submit(live_data, current_round):
    print('Submitting...')

    predict_output_path = f"live_predictions_{current_round}.csv"

    # make new dataframe with only the index (contains ids)
    predictions_df = live_data['id'].to_frame()
    # copy predictions into new dataframe
    predictions_df["prediction"] = live_data["prediction"].copy()
    predictions_df.to_csv(predict_output_path, index=False)

    print(f'submitting {predict_output_path}')
    napi.upload_predictions(predict_output_path, model_id=MODEL_ID)
    print('submission complete!')

    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()


#@functions_framework.http
#def hello_numerai(request):
#    main()

#    return 'Numerai Submission!'


def main():
    """ Download, train, predict and submit for this model """

    metric = 'val_payout'
    target = 'target'
    mode = 'max'
    model_base = LitTransformer
    project_name = f'Numerai_V43_wffull_nonblock'
    hp = ['attention_heads', 'dropout', 'hidden_dim', 'lr', 'n_hidden', 'batch_size', 'l1reg']
    params = Params(input_dim=None, hidden_dim=16, n_hidden=1, dropout=0.11457303675039694, seed=42, output_dim=1,
                    lr=0.0022323906533276124, batch_size=2, epochs=20, l2reg=None, l1reg=0.0030119644549417984,
                    opt='adam', attention_heads=4, gradient_clip_val=0, n_envs=None, slength=None, pos_encoding=None)
    params.criterion = spearman_correlation_loss
    patience_es = 5
    loop_params = LoopParameters.load(f'params_loop.pkl')
    loop_params.live_round = napi.get_current_round()

    #save_best_models(project_name, 'mr1ijjkp', params, 'k_fold_score', top_k=5)

    data, live_data, features = download_data()
    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()

    cv = WalkForwardFull(data, gap=4)
    for idx, walk in cv.get_walks():
        params.fold = None
    wf_block = False if isinstance(cv, WalkForwardFull) else True
    if cv.count > loop_params.train_i:
        train(project_name, params, 0, target, data, walk, features, model_base, metric, mode, patience_es, hp, wf_block, version='V1')
        train(project_name, params, 2, target, data, walk, features, model_base, metric, mode, patience_es, hp, wf_block, version='V1')
        train(project_name, params, 0, target, data, walk, features, model_base, metric, mode, patience_es, hp, wf_block, version='V2')
        train(project_name, params, 3, target, data, walk, features, model_base, metric, mode, patience_es, hp, wf_block, version='V2')
        loop_params.train_i += 1

    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()
    new_live_data = predict(0, live_data, features, model_base, params, project_name, metric, mode, hp)
    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()

    submit(new_live_data, loop_params.live_round)

    # "garbage collection" (gc) gets rid of unused data and frees up memory
    gc.collect()
    loop_params.dump(f'params_loop.pkl')


if __name__ == '__main__':

    """
    names = ['cyrus_model0', 'cyrus_model1', 'cyrus_model2', 'cyrus_model3', 'cyrus_model4', 'teager_model0', 'teager_model1', 'teager_model2', 'teager_model3', 'teager_model4']
    import itertools
    lengths=[]
    for i in range(len(names)):
        names_ = names[:i]
        all_comb = []
        for r in range(2, len(names_) + 1):
            all_comb.extend(list(itertools.combinations(names_, r)))
        lengths.append(len(all_comb))
    pd.Series(lengths).plot()
    """

    #main()

    # !pip install -q --no-deps numerai-tools
    from numerai_tools.scoring import numerai_corr, correlation_contribution, feature_neutral_corr
    from pylab import plt

    try:
        delete_files('lightning_logs')
        delete_files('numerai')
        delete_files('wandb')
    except:
        pass


    """
    napi = NumerAPI()
    napi.download_dataset("v4.1/train_int8.parquet")
    napi.download_dataset("v4.1/validation_int8.parquet")
    napi.download_dataset("v4.1/features.json")
    napi.download_dataset("v4.3/meta_model.parquet")
    napi.download_dataset("v4.3/train_benchmark_models.parquet")
    napi.download_dataset("v4.3/validation_benchmark_models.parquet")
    """

    train_bench = pd.read_parquet("v4.3/train_benchmark_models.parquet")
    val_bench = pd.read_parquet("v4.3/validation_benchmark_models.parquet")
    bench_name = 'v43_lgbm_teager20'
    bench = pd.concat([train_bench, val_bench[~val_bench[bench_name].isna()]])[['era', bench_name]]
    bench.columns = ['era', 'numerai_bench_model']

    #napi.download_dataset("v4.3/live_int8.parquet")
    # train = pd.read_parquet("v4.1/train_int8.parquet")
    # val = pd.read_parquet("v4.1/validation_int8.parquet")
    # data = pd.concat([train, val[~val['target'].isna()]])
    # data.to_parquet("v4.3/data_int8.parquet")

    # napi.download_dataset("v4.2/train_int8.parquet")

    feature_metadata = json.load(open("v4.3/features.json"))

    features = feature_metadata["feature_sets"]["small"]
    target = "target_teager_v4_60"
    data = pd.read_parquet("v4.3/data_int8.parquet", columns=["era"] + features + ["target"] + [target])
    data['id'] = data.index
    data.index.name = 'days'

    meta_model = pd.read_parquet(f"v4.3/meta_model.parquet", columns=['era', 'numerai_meta_model'])

    data = data.merge(meta_model, on=['id', 'era'], how='left')
    data = data.merge(bench, on=['id', 'era'], how='left')
    data.index = data.era.astype(int)
    # test = pd.read_parquet("v4.2/live_int8.parquet", columns=["era"] + feature_cols)
    data = data.loc[905: 1091]

    missing_idx = list(data[["target"] + [target]][data[["target"] + [target]].isna()[target]].index.unique())
    if len(missing_idx) > 0:
        print(f'missing index for {target}: {missing_idx}')
        data.dropna(inplace=True)

    """
    data.index = data.id
    data.index.name = 'id'
    per_era_corr = data[['era', 'numerai_bench_model', 'target']].dropna().groupby("era").apply(
        lambda x: numerai_corr(x[['numerai_bench_model']].dropna(), x["target"].dropna())
     )
    """

    del meta_model
    del train_bench
    del val_bench
    del bench

    metric = 'val_payout'
    mode = 'max'
    model_base = LitTransformer
    project_name = f'test'
    hp = ['attention_heads', 'dropout', 'hidden_dim', 'lr', 'n_hidden', 'batch_size', 'l1reg']
    params = Params(input_dim=None, hidden_dim=8, n_hidden=1, dropout=0.15241275848829344, seed=42, output_dim=1,
                    lr=0.002321695031074939, batch_size=2, epochs=1, l2reg=None, l1reg=0.009584595998367966,
                    opt='adam', attention_heads=4, gradient_clip_val=0, n_envs=None, slength=None, pos_encoding=None)
    params.criterion = spearman_correlation_loss
    patience_es = 5
    #params_instanciation(project_name, params, 0)
    #save_best_models(project_name, 'gc4tdrnn', params, 'k_fold_score', top_k=3)



    cv = WalkForwardFull(data, gap=4)
    submissions = []
    for idx, walk in cv.get_walks():
        params.fold = None

        train_env, valid_env = compute_envs_fit(target, data, walk, features, fold=idx)
        compute_fit((train_env, valid_env), model_base, project_name, metric, mode, params,
                    name_save=None, patience_es=patience_es, wandb_b=False, hp=hp,
                   fast_dev_run=False, wf_block=False)
        test_env = MarketEnv(data, features, None, walk.test.start, walk.test.end)
        y = compute_pred(test_env, model_base, params, project_name, metric, mode, hp)
        submission = pd.Series(y, index=test_env.data['id']).to_frame('prediction')
        submissions.append(submission)

    #submissions = pd.concat(submissions)




    sweep_config = {
        "method": "bayes",  # bayes

        "metric": {
            "name": 'k_fold_score',
            "goal": mode + 'imize'
        },
        "parameters": {
            "attention_heads": {
                'values': [1]
            },
            "dropout": {
                'max': 0.3,
                'min': 0.1
            },
            "hidden_dim": {
                'values': [8]
            },
            "lr": {
                'max': 0.01,
                'min': 0.0099
            },
            "n_hidden": {
                'values': [1]
            },
            "batch_size": {
                'values': [2]
            },
            "l1reg": {
                'max': 1e-2,
                'min': 1e-4
            }
        }
    }

    sweep_config = {
        "method": "bayes",  # bayes

        "metric": {
            "name": 'k_fold_score',
            "goal": mode + 'imize'
        },
        "parameters": {
            # "attention_heads": {
            #    'values': [1, 4]
            # },
            # "batch_size": {
            #    'values': [2, 4]
            # },
            "dropout": {
                'max': 0.3,
                'min': 0.1
            },
            "lr": {
                'max': 0.0099,
                'min': 0.0001
            },
             "n_hidden": {
                'values': [1, 2]
             },
            "l1reg": {
                'max': 0.01,
                'min': 0.001
            }
        }
    }

    cv = WalkForwardFull(data, gap=4)
    compute_cv_tuning(sweep_config, data, model_base, params, features, target, metric, mode, project_name,
                      n_trials=20, top_k=3, patience_es=patience_es, cv=cv)
