import os
import sys
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm

num_cores = 8

def compute_melspec(filename, outdir):
    wav = librosa.load(filename, sr=44100)[0]
    melspec = librosa.feature.melspectrogram(
        wav,
        sr=44100,
        n_fft=128*20,
        hop_length=347*2,
        n_mels=128,
        fmin=20,
        fmax=44100 // 2)
    logmel = librosa.core.power_to_db(melspec)
    np.save(outdir + os.path.basename(filename) + '.npy', logmel)

# Training and validation data
file_list = glob('./data/train/*.wav') + glob('./data/validate/*.wav')
os.makedirs('./data/logmelspec', exist_ok=True)
_ = Parallel(n_jobs=num_cores)(
        delayed(lambda x: compute_melspec(x, './data/logmelspec/'))(x)
        for x in tqdm(file_list))

# Eval data
file_list = glob('./data/audio-eval/*.wav')
os.makedirs('./data/logmelspec-eval', exist_ok=True)
_ = Parallel(n_jobs=num_cores)(
        delayed(lambda x: compute_melspec(x, './data/logmelspec-eval/'))(x)
        for x in tqdm(file_list))
