from sklearn import svm
from os import listdir
import joblib
import pickle
import numpy as np
from question1_1 import stft, padSpectogram, drawSpectrogram
from question2_1 import calcMFCC, showMFCC
from scipy.io import wavfile
import math
from sklearn.metrics import classification_report
import random

convertDict = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}

def CreateSpectrogramNoise():
	pickle_features = open("spectrogram.pickle","rb")
	spectrograms = pickle.load(pickle_features)
	finalSpectrograms = []
	classes = []
	noises = []
	for files in listdir("Dataset/_background_noise_"):
		print("IN")
		noises.append(wavfile.read("Dataset/_background_noise_" + "/" + files)[1])


	for i in spectrograms.keys():
		randomNoise = random.randint(0, 5)
		print(randomNoise)
		noiseAdder = random.randint(0,500)
		print(noiseAdder)
		if noiseAdder == 20:
			print("20")
			finalSpectrograms.append(spectrograms[i].flatten() + noises[randomNoise].flatten()[:15744])
		else:
			finalSpectrograms.append(spectrograms[i].flatten())
		classes.append(convertDict[i.split('-')[0]])
	clf = svm.LinearSVC()
	clf.fit(finalSpectrograms, classes)
	joblib.dump(clf, 'svm_spectrogram_noise.pickle')
# CreateSpectrogramNoise()

def CreateMFCCNoise():
	pickle_features = open("MFCC_all.pickle","rb")
	mfccs = pickle.load(pickle_features)
	finalMFCC = []
	classes = []
	noises = []
	for files in listdir("Dataset/_background_noise_"):
		print("IN")
		noises.append(wavfile.read("Dataset/_background_noise_" + "/" + files)[1])

	for i in mfccs.keys():
		randomNoise = random.randint(0, 5)
		print(randomNoise)
		noiseAdder = random.randint(0,500)
		print(noiseAdder)
		if noiseAdder == 20:
			print("20")
			finalMFCC.append(np.nan_to_num(mfccs[i].flatten() + noises[randomNoise].flatten()[:2214]))
		else:
			finalMFCC.append(np.nan_to_num(mfccs[i].flatten()))
		classes.append(i.split('-')[0])
	clf = svm.LinearSVC()
	clf.fit(finalMFCC, classes)
	joblib.dump(clf, 'svm_mfcc_noise.pickle')
# CreateMFCCNoise()

def createSpectrogramSVM():
	pickle_features = open("spectrogram_N.pickle","rb")
	spectrograms = pickle.load(pickle_features)
	finalSpectrograms = []
	classes = []
	for i in spectrograms.keys():
		print("II")
		finalSpectrograms.append(spectrograms[i].flatten())
		classes.append(convertDict[i.split('-')[0]])
	print(classes)
	clf = svm.LinearSVC()
	clf.fit(finalSpectrograms, classes)
	joblib.dump(clf, 'svm_spectrogram_N.pickle')
# createSpectrogramSVM()

def createMFCCSVM():
	pickle_features = open("MFCC_inbuilt.pickle","rb")
	mfccs = pickle.load(pickle_features)
	finalMFCC = []
	classes = []
	for i in mfccs.keys():
		newMFCC = mfccs[i]
		finalMFCC.append(np.nan_to_num(newMFCC).flatten())
		classes.append(convertDict[i.split('-')[0]])
	print(classes)
	print(len(classes))	
	print(len(finalMFCC))
	# showMFCC(mfccs[i])
	clf = svm.LinearSVC()
	clf.fit(finalMFCC, classes)
	joblib.dump(clf, 'svm_mfcc_inbuilt.pickle')
# createMFCCSVM()

def predictViaSpectrogram(path,clf):
	sampleRate, audio = wavfile.read(path)
	X = stft(audio, sampleRate, 256)
	X = np.abs(padSpectogram(X))
	# drawSpectrogram(X)	
	return clf.predict([X.flatten()])[0]
# clf = joblib.load('svm_spectrogram_new.pickle')
# print(predictViaSpectrogram( "Dataset/training/four/0cd323ec_nohash_1.wav",clf))

def predictViaMFCC(path,clf):
	mfcc = calcMFCC(path)
	newMFCC = mfcc
	return clf.predict([np.nan_to_num(newMFCC).flatten()])[0]
# clf = joblib.load('svm_spectrogram_new.pickle')
# # print(predictViaMFCC('Dataset/training/zero/004ae714_nohash_0.wav',clf))	

clf = joblib.load('svm_mfcc_noise.pickle')
folderPath = "Dataset/validation"
folders = listdir(folderPath)
results = []
actual = []
counter2 = 0
for folder in folders:
    filePath = folderPath + "/" + folder
    files = listdir(filePath)    
    for i in files:
    	path = filePath +"/" + i
    	result = predictViaMFCC(path, clf)
    	results.append(result)
    	actual.append(folder)
    	print(path +"     "+ str(counter2) +"    "+str(result == folder))
    	print(result)
    	counter2 +=1

print(classification_report(actual, results))
# # print(counter)
