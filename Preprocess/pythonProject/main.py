import functools
import itertools
from glob import glob
from tqdm import tqdm
import multiprocessing as mp

from audio_preprocess import *

ROOT_PATH = r"D:\Users\GraduationDesign"
SRC_PATH = ROOT_PATH + r"\Audio"
TGT_PATH = ROOT_PATH + r"\AudioProcessed"

dialects = []

paths = glob(SRC_PATH + r"\*")
for path in paths:
    dialects.append(path.split('\\')[4])


# 耗时1个小时
if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        for dialect in tqdm(dialects,
                            desc="Total missions",
                            total=len(dialects),
                            position=0):
            src_paths = glob(SRC_PATH + "\\" + dialect + r"\*\*\*\*\*.wav")
            trt = TGT_PATH + "\\" + dialect
            # it = itertools.product(src_paths,trt)  # 迭代器生成
            # pool.map_async(audio_preprocess, it)

            f = functools.partial(audio_preprocess, trt_path=trt)  # 固定一个变量，产生新方法
            for _ in tqdm(pool.imap_unordered(f, src_paths),
                          desc=dialect,
                          total=len(src_paths),
                          position=1):
                pass
        # 使用with as 自动执行以下方法
        # pool.close()
        # pool.join()
