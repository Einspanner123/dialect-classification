import functools
import pickle
import time

import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from python_speech_features import mfcc
from tqdm import tqdm

class AudioUtils:
    def __init__(self, slice_length, min_length=None, mfcc_dim=13, sr=16000):
        self.slice_length = slice_length
        if min_length is None:
            self.min_length = slice_length
        self.mfcc_dim = mfcc_dim
        self.sr = sr

    def load_and_trim(self, path):
        """
        语音分片
        """
        audio, sr = librosa.load(path, sr=self.sr)
        energy = librosa.feature.rms(y=audio)  # 调用librosa中的rmse直接对音频每个帧进行计算得到均方根能量
        frames = np.nonzero(energy >= (np.max(energy) / 5))  # 返回大于五分之一最大能量的能量索引
        indices = librosa.core.frames_to_samples(frames)[1]  # librosa.core.frames_to_samples():将帧索引转换为音频样本索引，0为样本开始，1为样本结束

        audio = audio[indices[0]:indices[-1]] if len(indices) else []  # audio中存放去除头尾空白音频部分

        slices = []
        for i in range(0, len(audio), self.slice_length):  # 对长语音片段分片，shape数组大小
            s = audio[i: i + self.slice_length]
            if len(s) < self.min_length:
                break
            slices.append(s)

        return audio, slices

    def visualize(self, path):
        """
        可视化语音
        """
        # 画出原始信号在时域中具有的形式
        audio, slices = self.load_and_trim(path)
        sr = librosa.get_samplerate(path)
        print('Duration: %.2fs' % (len(audio) / sr))
        plt.figure(figsize=(12, 3))
        plt.plot(np.arange(len(audio)), audio)
        plt.title('Raw Audio Signal')
        plt.xlabel('Time')
        plt.ylabel('Audio Amplitude')
        plt.show()

        # 获取13维的 MFCC 特征
        feature = mfcc(audio, sr, numcep=self.mfcc_dim)
        print('Shape of MFCC:', feature.shape)

        # 画出MFCC特征图的频谱图
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(feature, cmap=plt.cm.jet, aspect='auto')
        plt.title('Normalized MFCC')
        plt.ylabel('Time')
        plt.xlabel('MFCC Coefficient')
        plt.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_xticks(np.arange(0, 13, 2), minor=False)
        plt.show()

    def separate_data_from_path(self, train_files):
        func_mfcc = functools.partial(librosa.feature.mfcc, samplerate=self.sr, numcep=self.mfcc_dim)

        # 打印日期时间
        print(time.strftime("%y-%m-%d %T", time.localtime()))

        # 对训练集语音分片后各片段进行MFCC特征提取以及与标签对应
        for i in tqdm(range(len(train_files))):
            path = train_files[i]
            audio, slices = load_and_trim(path)
            lengths.append(len(audio) / sr)

            for s in slices:
                X_train.append(func_mfcc(s))
                Y_train.append(labels['train'][i])

        # 打印日期时间
        print(time.strftime("%y-%m-%d %T", time.localtime()))

        # 对测试集语音分片后各片段进行MFCC特征提取以及与标签对应
        for i in tqdm(range(len(valid_files))):
            path = valid_files[i]
            audio, slices = load_and_trim(path)
            lengths.append(len(audio) / sr)
            for s in slices:
                X_valid.append(func_mfcc(s))
                Y_valid.append(labels['valid'][i])



    @staticmethod
    def save_data(data, path):
        with open(path, 'wb') as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load_data(path):
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        return data['X_train'], data['Y_train'], data['X_valid'], data['Y_valid'], data['lengths']
