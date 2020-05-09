from os import listdir
import numpy as np
import scipy
from scipy.io import wavfile
from scipy import signal
import pylab
import scipy.fftpack as fft
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import librosa
import math
import pickle

windowSize = 256
overlapSize = int(windowSize/2)


def padSpectogram(spectrogram):
    paddedSpectrogram = np.zeros((128, 123), dtype=np.complex64, order='F')
    paddedSpectrogram[:spectrogram.shape[0],: spectrogram.shape[1]] = spectrogram
    return np.abs(paddedSpectrogram)

def stft(audio, sampleRate,windowSize):
    overlapSize = int(windowSize/2)
    frames = []
    window = np.hanning(windowSize)
    for i in range(0, (len(audio) - windowSize), overlapSize):
        frames.append(np.fft.fft(window * audio[i:i + windowSize]))
    stftFrames = np.array(frames)
    spectrogram = np.transpose(np.square(np.abs(stftFrames[:, :int(np.shape(stftFrames)[1]/2)])))
    # spectrogram = np.square(np.abs(signal.spectrogram(audio, sampleRate)[2]))
    print(spectrogram.shape)
    spectrogram = padSpectogram(spectrogram)
    # print(spectrogram.shape)
    return spectrogram

def getFilterScales(windowSize, sampleRate):
    minFreq = 0
    maxFreq = sampleRate / 2
    minMelFreq = 0
    maxMelFreq = (2595 * np.log10(1 + (maxFreq) / 700)) 
    numMelFilter = 20
    mels = np.linspace(minMelFreq, maxMelFreq, num = numMelFilter)
    freqs = (700 * (10**(mels / 2595) - 1))
    return np.floor((windowSize + 1) / sampleRate * freqs).astype(int)

def getFilters(windowSize,sampleRate):
    filterPoints = getFilterScales(windowSize, sampleRate)
    filters = np.zeros((len(filterPoints) - 2,int(windowSize / 2)))
    for i in range(1, len(filterPoints) - 1):
        fmLeft = int(filterPoints[i - 1])   # left
        fmCentre = int(filterPoints[i])             # center
        fmRight = int(filterPoints[i + 1])    # right

        for k in range(filterPoints[0], filterPoints[len(filterPoints) - 1]):
            if k in range(fmLeft, fmCentre):
                filters[i - 1, k] = (k - filterPoints[i - 1]) / (filterPoints[i] - filterPoints[i - 1])

            if k in range(fmCentre, fmRight):
                filters[i - 1, k] = (filterPoints[i + 1] - k) / (filterPoints[i + 1] - filterPoints[i])
    return filters

def showMFCC(mfcc):
    plt.imshow(librosa.core.amplitude_to_db(np.abs(mfcc)), aspect="auto")
    plt.title('MFCC Plot')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def calcMFCC(path):
    sampleRate, audio = wavfile.read(path)
    audioSTFT = stft(audio, sampleRate,windowSize)
    filters = getFilters(windowSize, sampleRate)
    audioFilterApplied = np.dot(filters, audioSTFT)
    # print(audioFilterApplied)
    audioDB = 20.0 * np.log10(audioFilterApplied)
    mfcc = np.abs(fft.dct(audioDB, type=2, axis=0, norm='ortho'))
    return mfcc

# import time
# start = time.time()
# print(start)


# mfccDict = {}
# folderPath = "Dataset/training"
# folders = listdir(folderPath)
# counter = 0
# for folder in folders:
#     filePath = folderPath + "/" + folder
#     files = listdir(filePath)    
#     for i in files:
#         path = filePath +"/" + i
#         print(path +"     "+ str(counter))
#         X = calcMFCC(path)
#         mfccDict[folder+"-"+i] = X
#         # showMFCC(X)
#         counter += 1
#     pickle_features = open("MFCC_inbuilt.pickle","wb")
#     pickle.dump(mfccDict,pickle_features)

# pickle_features = open("MFCC.pickle","rb")
# dica = pickle.load(pickle_features)
# X = dica['zero-0ff728b5_nohash_1.wav']
# print(X.shape)
# showMFCC(X)
# for i in dica.keys():
#     print(dica[i].shape)

# mfcc = calcMFCC("Dataset/training/four/0cd323ec_nohash_1.wav")    
# showMFCC(mfcc)
# audio, sampleRate = librosa.load("Dataset/training/four/0cd323ec_nohash_1.wav")
# mfcc = librosa.feature.mfcc(audio, sampleRate, n_mfcc= 12)#
# showMFCC(mfcc)
# end = time.time()
# print(end - start)
