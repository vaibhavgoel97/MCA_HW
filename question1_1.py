from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import pylab
import numpy as np
import scipy.fftpack as fft
from os import listdir
import pickle
import librosa

def drawSpectrogram(spec):
    plt.imshow(librosa.core.amplitude_to_db(spec), aspect="auto")
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

filename = 'Dataset/training/zero/0bde966a_nohash_1.wav'
def inbuiltSpectrogram(filename):
	fs, data = wavfile.read(filename)
	frequencies, times, spectrogram = signal.spectrogram(data, fs);print("------------------");print(spectrogram.shape);drawSpectrogram(spectrogram)
# inbuiltSpectrogram(filename)

# def dft(x):
#     x = np.array(x, dtype=int)
#     N = x.shape[0]
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     M = np.exp(-2j * np.pi * k * n / N)
#     return np.dot(M, x)

# def fft(x):
#     x = np.array(x, dtype=float)
#     N = x.shape[0]

#     if N <= 2:
#         return dft(x)
#     else:
#         X_even = fft(x[::2])
#         X_odd = fft(x[1::2])
#         terms = np.exp(-2j * np.pi * np.arange(N) / N)
#         return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
#                                X_even + terms[int(N/2):] * X_odd])

def stft(audio, sampleRate, windowSize):
    overlapSize = int(windowSize/2)
    frames = []
    window = np.hanning(windowSize)
    for i in range(0, (len(audio) - windowSize), overlapSize):
        frames.append(fft(window * audio[i:i + windowSize]))
    stftFrames = np.array(frames)
    spectrogram = np.transpose(np.square(np.abs(stftFrames[:, :int(np.shape(stftFrames)[1]/2)])))
    return spectrogram

# import time
# start = time.time()
# print(start)

def padSpectogram(spectrogram):
    paddedSpectrogram = np.zeros((128, 123), dtype=np.complex64, order='F')
    paddedSpectrogram[:spectrogram.shape[0],: spectrogram.shape[1]] = spectrogram
    return paddedSpectrogram

# pickle_features = open("spectrogram_N.pickle","rb")
# dica = pickle.load(pickle_features)
# X = dica['zero-0ff728b5_nohash_1.wav']
# drawSpectrogram(X)
# for i in dica.keys():
#     print(dica[i].shape)

# spectoDict = {}
# folderPath = "Dataset/training"
# folders = listdir(folderPath)
# counter = 0
# for folder in folders:
#     filePath = folderPath + "/" + folder
#     files = listdir(filePath)    
#     for i in files:
#         path = filePath +"/" + i
#         print(path +"     "+ str(counter))
#         sampleRate, audio = wavfile.read(path)
#         X = stft(audio, sampleRate, 256)
#         X = np.abs(padSpectogram(X))
#         print(X.shape)
#         T = len(audio)/sampleRate
#         spectoDict[folder+"-"+i] = X
#         # drawSpectrogram(X)
#         counter += 1
#     pickle_features = open("spectrogram.pickle","wb")
#     pickle.dump(spectoDict,pickle_features)
# #         # np.save("spectrogram_files/"+ folder + i[:-4], X)
        

# sampleRate, audio = wavfile.read('Dataset/training/zero/004ae714_nohash_0.wav')
# X = stft(audio, sampleRate, 256)
# print(X.shape)
# X = np.abs(padSpectogram(X))
# print(X.shape)
# drawSpectrogram(X)
# end = time.time()
# print(end - start)