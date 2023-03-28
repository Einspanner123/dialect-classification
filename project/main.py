# -*- coding:utf-8 -*-
import os
import time
# 设置环境变量（放在import tf前才生效）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"

import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow._api.v2.compat.v1 as tf
# tf.enable_eager_execution()
import numpy as np
from sklearn.utils import shuffle
from glob import glob
import pickle
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv1D, Dropout, LSTM
from sklearn.model_selection import train_test_split

from utils import AudioUtils
from layers import *
from preprocess import *
from model import *

# data_path = r'D:\Users\GraduationDesign\AudioProcessed'
data_path = r"../../autodl-tmp/AudioProcessed"




# 设置GPU
from warnings import simplefilter

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
simplefilter(action='ignore', category=FutureWarning)

# 获取指定路径下所有的pcm文件
files = glob(data_path + r"\*\*.wav")

train_files, valid_files = train_test_split(files, test_size=0.99, random_state=40)
train_files, valid_files = train_test_split(train_files, test_size=0.2, random_state=40)
print("train data:%d, valid_data:%d" % (len(train_files), len(valid_files)))


def main():


    history = model.fit_generator(
        generator=batch_generator(X_train, Y_train),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=batch_generator(X_valid, Y_valid),
        validation_steps=len(X_valid) // batch_size,
        callbacks=[lr_decay]
    )


if __name__ == '__main__':
    main()
    # print(tf.test.is_gpu_available())
