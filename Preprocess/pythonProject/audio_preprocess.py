import os
from scipy.ndimage.morphology import binary_dilation
import librosa
import librosa.display
import numpy as np
import struct
import webrtcvad  # VAD
import soundfile as sf
import noisereduce as nr

INT16_MAX = (1 << 15) - 1


# 𝑣_𝑏𝑖𝑎𝑠𝑒𝑑𝑡=𝑣𝑡/(1−𝛽𝑡)
# 滑动平均计算
def moving_average(array, width):
    # 拼接 bool 二值化
    # width 执行滑动平均平滑时，帧的平均数。
    # 该值越大，VAD变化必须越大才能平滑。
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    # 一维数组累加
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]

    return ret[width - 1:] / width


def write_wav(path, wav, sr):
    sf.write(path, wav.astype(np.float32), samplerate=sr, subtype='PCM_24')


def remove_noise(wav, sr):
    # 去除噪音
    wav = nr.reduce_noise(y=wav, sr=sr)
    return wav


def remove_blanks(wav, sr):
    # 计算语音检测窗口大小  //为整除30秒X16000=总帧长
    samples_per_window = (30 * 16000) // 1000

    # wav, _ = librosa.effects.trim(wav) # 去除首尾沉默

    # 修剪音频的结尾，使其具有窗口大小的倍数。使wav的长度能被 samples_per_window整除
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # 浮点数波形转换为16位单声道PCM  *：接收到的参数会形成一个元组，**：接收到的参数会形成一个字典。如下代码。
    # webrtcvad 的 is_speech 接收的是buf 所以这里需要转换
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * INT16_MAX)).astype(np.int16))

    # 执行语音激活检测
    voice_flags = []
    # 这里共有三种帧长可以用到，分别是80/10ms，160/20ms，240/30ms。其它采样率
    # 的48k，32k，24k，16k会重采样到8k来计算VAD。之所以选择上述三种帧长度，是因为语
    # 音信号是短时平稳信号，其在10ms~30ms之间可看成平稳信号，高斯马尔科夫等比较
    # 的信号处理方法基于的前提是信号是平稳的，在10ms~30ms，平稳信号处理方法是可
    # 以使用的。
    # 　　从vad的代码中可以看出，实际上，系统只处理默认10ms,20ms,30ms长度的数据，
    # 其它长度的数据没有支持，笔者修改过可以支持其它在10ms-30ms之间长度的帧长度
    # 发现也是可以的。
    # 　　vad检测共四种模式，用数字0~3来区分，激进程度与数值大小正相关。
    # 0: Normal，1：low Bit rate， 2：Aggressive；3：Very Aggressive 可以根据实际的使用
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        # append 进来的都是Boolean  这里以samples_per_window*2 的长度去检测是否为人声
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2: window_end * 2], sample_rate=sr))
    voice_flags = np.array(voice_flags)

    # 滑动平均计算
    audio_mask = moving_average(voice_flags, 8)
    # 将平均数四舍五入 转bool
    audio_mask = np.round(audio_mask).astype(np.bool_)
    # 扩张浊音区 使用多维二元膨胀 是数学形态学的方法 类似opencv 也有开闭运算 腐蚀膨胀
    audio_mask = binary_dilation(audio_mask, np.ones(6 + 1))
    # 使其与 wav一样大小
    audio_mask = np.repeat(audio_mask, samples_per_window)
    # 通过这个遮罩扣掉没有声音那部分
    res_wav = wav[audio_mask != 0]

    return res_wav


# 预处理步骤，path末尾统一不带反斜杠
def audio_preprocess(src_path, trt_path):
    wav, sr = librosa.load(src_path, sr=None)
    wav_name = src_path.split("\\")[-1]

    '''
    经测试后，先去噪音后去空白得到的音频去噪效果最好，但会损失一些数据
    '''
    # 音频处理步骤
    wav = remove_noise(wav, sr)
    wav = remove_blanks(wav, sr)

    # 经过测试，该方法耗时少，约三分之一
    if not os.path.exists(trt_path):
        os.makedirs(trt_path, exist_ok=True)

    # 判断是否存在对应文件
    trt_path = trt_path + "\\" + wav_name
    if not os.path.exists(trt_path):
        write_wav(trt_path, wav, sr)
