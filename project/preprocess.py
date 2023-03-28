import functools
import time

import librosa.feature
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import pickle


class Preprocess:
    def __init__(self, x_train, x_valid, y_train, y_valid):
        self.X_train = x_train
        self.X_valid = x_valid
        self.Y_train = y_train
        self.Y_valid = y_valid
        self.lengths = []

    def label_to_category(self, save_path=None):
        le = LabelEncoder()
        Y_train = le.fit_transform(self.Y_train)  # 用0，1，2，3...表示不同类别
        Y_valid = le.transform(self.Y_valid)

        if save_path is not None:
            class2id = {c: i for i, c in enumerate(le.classes_)}  # 标签与数字之间对应关系
            id2class = {i: c for i, c in enumerate(le.classes_)}
            with open('resources.pkl', 'wb') as fp:
                pickle.dump([class2id, id2class], fp)

        num_class = len(le.classes_)
        Y_train = to_categorical(Y_train, num_class)  # 将类别向量转化为独热编码的矩阵向量
        Y_valid = to_categorical(Y_valid, num_class)

        return Y_train, Y_valid

    def mfcc_normalize(self):
        # 对 mfcc特征进行归一化
        samples = np.vstack(self.X_train)  # vstack()将数据整合到一个数组中
        mfcc_mean = np.mean(samples, axis=0)  # 求均值，axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        mfcc_std = np.std(samples, axis=0)  # 求标准差

        X_train = [(x - mfcc_mean) / (mfcc_std + 1e-14) for x in self.X_train]
        X_valid = [(x - mfcc_mean) / (mfcc_std + 1e-14) for x in self.X_valid]

        max_len = np.max([x.shape[0] for x in X_train + X_valid])
        X_train = pad_sequences(X_train, max_len, 'float32', padding='post', value=0.0)
        X_valid = pad_sequences(X_valid, max_len, 'float32', padding='post', value=0.0)

        return X_train, X_valid


    def process(self):
        """
        Returns ->
        X_train, Y_train, X_valid, Y_valid
        """
        X_train, Y_train = self.mfcc_normalize()
        X_valid, Y_valid = self.label_to_category()
        return X_train, Y_train, X_valid, Y_valid


