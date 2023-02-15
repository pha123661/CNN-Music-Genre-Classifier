import warnings
import os
import librosa
import numpy as np
import urllib
import torch
from moviepy.editor import AudioFileClip
from pytube import YouTube
from CNN_Train_Test_Plot import *
from Hyper_parameters import HyperParams

warnings.filterwarnings("ignore")


def feature_extraction(filename, debug=False):
    def melspectrogram():
        y, sr = librosa.load(filename, HyperParams.sample_rate)
        S = librosa.stft(y, n_fft=HyperParams.fft_size,
                         hop_length=HyperParams.hop_size, win_length=HyperParams.win_size)

        mel_basis = librosa.filters.mel(
            HyperParams.sample_rate, n_fft=HyperParams.fft_size, n_mels=HyperParams.num_mels)
        mel_S = np.dot(mel_basis, np.abs(S))
        mel_S = np.log10(1+10*mel_S)
        mel_S = mel_S.T

        return mel_S

    def resize_array(array, length):
        resized_array = np.zeros((length, array.shape[1]))
        if array.shape[0] >= length:
            resize_array = array[:length]
        else:
            resized_array[:array.shape[0]] = array
        return resize_array

    feature = melspectrogram()
    if debug:
        feature = resize_array(feature, HyperParams.feature_length)
    num_chunks = feature.shape[0]/HyperParams.num_mels
    return np.split(feature, num_chunks)


if __name__ == '__main__':
    print("Available genres:", HyperParams.genres)
    Model = torch.load("Demo_CNN_model.pth",
                       map_location=torch.device('cpu')).eval()
    while True:
        link = input("Please enter YouTube link: ")
        yt = YouTube(link)
        while True:
            try:
                name = yt.streams.filter(only_audio=True).first().download()
                wav_name = name.split(".")[0]+".wav"
                break
            except urllib.error.HTTPError:
                print("Timeout, restarting download")
        print('Finish downloading: "'+name.split("\\")[-1]+'"')
        print("Generating Mel spectrogram...")
        my_audio_clip = AudioFileClip(name)
        my_audio_clip.write_audiofile(wav_name, logger=None)
        data_chuncks = feature_extraction(wav_name)
        data_chuncks = [d for d in data_chuncks if d.shape == (128, 128)]
        if not len(data_chuncks):
            data_chuncks = feature_extraction(wav_name, debug=True)
            data_chuncks = [d for d in data_chuncks if d.shape == (128, 128)]
        os.unlink(name)
        os.unlink(wav_name)
        rst = Model(torch.Tensor(data_chuncks)).detach().numpy()
        rst = np.argmax(np.bincount(np.argmax(rst, axis=1)))
        print("********************************")
        print("AI thinks it's a " + HyperParams.genres[rst], "music")
        print("********************************")
        if input("Try another song? [Y/n]: ") in ["n", "N", "No", "no"]:
            break
