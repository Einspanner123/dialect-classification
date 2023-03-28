from numpy.random import shuffle
from sklearn.utils import shuffle
from tensorflow.keras.layers import BatchNormalization, Activation, Multiply, Add, GlobalMaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
from layers import *

mfcc_dim = 13  # 通道数为13
sr = 16000  # 采样率为16000
min_length = sr * 1  # 最小1s
slice_length = sr * 3  # 最大3s
epochs = 30
num_blocks = 3
filters = 128
drop_rate = 0.2
batch_size = 16


def batch_generator(x, y, batch_size):
    offset = 0
    while True:
        offset += batch_size

        if offset == batch_size or offset >= len(x):
            x, y = shuffle(x, y)  # shuffle():对多个等长列表打乱，并且保证一一对应关系不变
            offset = batch_size

        X_batch = x[offset - batch_size: offset]
        Y_batch = y[offset - batch_size: offset]
        yield X_batch, Y_batch


class WaveNet:
    def __int__(self):
        self.X = Input(shape=(None, mfcc_dim), dtype='float32')
        h0 = activation(batch_normalization(conv1d(self.X, filters, 1, 1)), 'tanh')
        shortcut = []
        for i in range(num_blocks):
            for r in [1, 2, 4, 8, 16]:
                h0, s = res_block(h0, filters, 7, r)
                shortcut.append(s)

        h1 = activation(Add()(shortcut), 'relu')
        h1 = activation(batch_normalization(conv1d(h1, filters, 1, 1)), 'relu')  # batch_size, seq_len, filters
        h1 = batch_normalization(conv1d(h1, 7, 1, 1))  # batch_size, seq_len, num_class
        h1 = Dropout(drop_rate)(h1)
        h1 = GlobalMaxPooling1D()(h1)  # batch_size, num_class
        self.Y = activation(h1, 'softmax')

    def get_optimizer(self):
        return Adam(lr=0.01, clipnorm=5)  # 原 0.01

    def get_model(self):
        model = Model(inputs=self.X, outputs=self.Y)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])  # 原loss = 'categorical_crossentropy'
        return model

    @staticmethod
    def get_callback(self):
        # checkpointer = ModelCheckpoint(filepath='WaveNet_model', verbose=1)
        lr_decay = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=1, min_lr=0.000, verbose=1)
        return lr_decay
