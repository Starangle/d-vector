import librosa
import numpy as np

def extract(file):
    frame_length=2048
    hop_length=512

    y,sr=librosa.load(file)
    rms=librosa.feature.rms(y=y,frame_length=frame_length,hop_length=hop_length)
    mfcc=librosa.feature.mfcc(y=y,frame_length=frame_length,hop_length=hop_length)

    return np.concatenate((rms,mfcc),axis=0)