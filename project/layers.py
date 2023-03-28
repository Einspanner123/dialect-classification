from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Input, Conv1D, Dropout, LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Activation, Multiply, Add, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam


def conv1d(inputs, filters, kernel_size, dilation_rate):
    return Conv1D(filters=filters,
                  kernel_size=kernel_size,
                  strides=1,
                  padding='causal',  # 因果卷积
                  activation=None,
                  dilation_rate=dilation_rate)(inputs)


# 批量标准化
def batch_normalization(inputs):
    return BatchNormalization()(inputs)


def activation(inputs, activation_function):
    return Activation(activation_function)(inputs)


def res_block(inputs, filters, kernel_size, dilation_rate):
    hf = activation(batch_normalization(conv1d(inputs, filters, kernel_size, dilation_rate)), 'tanh')
    hg = activation(batch_normalization(conv1d(inputs, filters, kernel_size, dilation_rate)), 'sigmoid')
    h0 = Multiply()([hf, hg])

    ha = activation(batch_normalization(conv1d(h0, filters, 1, 1)), 'tanh')
    hs = activation(batch_normalization(conv1d(h0, filters, 1, 1)), 'tanh')

    return Add()([ha, inputs]), hs  # 残差层输出和跳连接层输出
