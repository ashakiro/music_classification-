import math
import csv
import os
import copy
import scipy
import random
import librosa
import itertools
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scikits.audiolab
from scikits.audiolab import Sndfile
import scipy.io.wavfile
from random import shuffle
from sklearn import svm, linear_model, datasets, decomposition, linear_model
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc
from python_speech_features import logfbank

############Plan of work############
def program():
    createFeatures()
	crossValidate()

#Create CSV for each feature
def createFeatures():
# 661794 - frames per song
	createFeature(FFTComponents,'FFT')
	createFeature(createMFCC,'MFCC')

##########Functions for working with data###########

#read .au
#return data
def readAUFile(path):
	f = Sndfile(path, 'r')
	data = f.read_frames(f.nframes)
	return data

#read csv
#return pandas dataframe
def readCSVToDF(csv):
	data = pd.read_csv(csv)
	return data

#read fft data
#return csv features
def plainDataToCSV(datalist,name):
	with open(name, 'wb') as myfile:
		wr = csv.writer(myfile)
		for row in datalist:
			wr.writerow(row)


#read list of lists
#return csv
def listToCSV(datalist,name):
	with open(name, 'wb') as myfile:
		wr = csv.writer(myfile)
		indexValues = range(0,len(datalist[0])-1)
		indexValues.append("Class")
		wr.writerow(indexValues)
		for row in datalist:
			wr.writerow(row)


#read path
#return file names
def getTotalDataset():
	fileTree = {}
	genres = os.listdir("genres")
	for genre in genres:
		if genre not in fileTree.keys():
			fileTree[genre] = []
		samples = os.listdir("genres/"+genre)
		fileTree[genre] = samples
	return fileTree

##########Functions for working with features###########

#read fft transform (first 1000)
#return as a feature
def FFTComponents(data):
	data = abs(scipy.fft(data)[:1000])
	return data

#read mfcc transform
#return averages across the frames in the range from 10% to 90%
def createMFCC(data):
	ceps, mspec, spec = mfcc(data)
	num_ceps = len(ceps)
	reducedCeps = ceps[int(num_ceps*0.10):int(num_ceps*0.90)]
	x = np.mean(reducedCeps,axis=0)
	return x

#read feature function
#return csv of feature function and validation data
def createFeature(featureFunction,name):
	FILE_PATH = "genres/"
	outputDataset = []
	dataset = getTotalDataset()
	count = 0
	for key in dataset.keys():
		for row in dataset[key]:
			data = [666,667]
			if count not in data:
				row = np.append(featureFunction(readAUFile(FILE_PATH+key+"/"+row)),mapGenreToInt(key))
				if math.isnan(float(row[1])):
					print count
				else:
					outputDataset.append(row)
			count+=1
	listToCSV(outputDataset,name+".csv")
	createTestDataFeatureCopy(featureFunction,name)


#read feature function
#return data set with it as csv
def createTestDataFeatureCopy(featureFunction,name):
	FILE_PATH = "test/"
	testData = os.listdir(FILE_PATH)
	results = []
	first = True
	for file in testData:
		if first:
			row = featureFunction(readAUFile(FILE_PATH+file))
			indexList = ["id"] + range(0,len(row))
			results.append(indexList)
			results.append(np.append([file],row))
			first = False
		else:
			row = featureFunction(readAUFile(FILE_PATH+file))
			results.append(np.append([file],row))
	plainDataToCSV(results,name+"_TEST.csv")

#dictionary of genres
def genres():
	genres = {
		"metal":10,
		"classical":9,
		"blues":8,
		"country":7,
		"disco":6,
		"hiphop":5,
		"jazz":4,
		"pop":3,
		"reggae":2,
		"rock":1
	}
	return genres

#read key
#return genre
def keyToGenre(key):
	reverseGenres = {v: k for k, v in genres().iteritems()}
	return reverseGenres[key]

#return value in dictionary
def mapGenreToInt(key):
	genreData = genres()
	return genreData[key]

#read feature sets (need for PCA)
#return combined set
def loadValidationFS(featureSet,normalize=False,PCA=False,PCA_N=0):
	dfset = []
	getClass = []
	for i in featureSet:
		df = readCSVToDF(i+".csv")
		getClass = df['Class']
		df = df.drop('Class',axis=1)
		dfset.append(df)
	df = pd.concat(dfset,axis=1)
	df.columns = range(0,len(pd.concat(dfset,axis=1).columns))
	if normalize:
		for i in df.columns.values:
	  		if i != 'Class':
	  			df[i] = ( df[i] - sum(df[i])/len(df) ) / (np.std(df[i]))
	if PCA:
		pca = decomposition.PCA(n_components=PCA_N) # this looks to be a pretty sweet spot for SVM
		pca.fit(df)
		df = pd.DataFrame(pca.transform(df))
	df['Class'] = getClass
	return df

 ##########Functions for classifiers##########

 #Support Vector Machine
def SVM(train,test):
 	train = train.fillna(method='backfill')
 	model = svm.SVC(kernel='linear',C=1.0,decision_function_shape='ovr')
 	collength = train.shape[1]
 	indexFetch = range(0,collength-2)
 	indexValues = train[indexFetch]
 	trainingSet = train["Class"]
 	trainFeatures = indexValues.values.tolist()
 	trainLabels = trainingSet.values.tolist()
 	tSVM = model.fit(trainFeatures,trainLabels)
 	features = test[indexFetch]
 	Z = tSVM.predict(features)
 	test["prediction"] = Z
 	return test


 #Logisitic Regression
def LG(train,test):
 	model = linear_model.LogisticRegression(penalty='l2',C=1e25)
 	collength = train.shape[1]
 	indexFetch = range(0,collength-2)
 	indexValues = train[indexFetch]
 	trainingSet = train["Class"]
 	trainFeatures = indexValues.values.tolist()
 	trainLabels = trainingSet.values.tolist()
 	logreg = model.fit(trainFeatures,trainLabels)
 	features = test[indexFetch]
 	Z = logreg.predict(features)
 	test["prediction"] = Z
 	return test

#############Functions for results###########

def crossValidate():
	print "Accuracy and confusion matrix"
	print "Only MFCC"
	printTest(SVM,['MFCC'],True,False,1000)
	printTest(LG,['MFCC'],True,False,1000)
	print "Only FFT"
	printTest(SVM,['FFT'],True,False,1000)
	printTest(LG,['FFT'],True,False,1000)
	print "Only MFCC with PCA"
	printTest(SVM,['MFCC'],True,True,13)
	printTest(LG,['MFCC'],True,True,13)
	print "Only FFT with PCA"
	printTest(SVM,['FFT'],True,True,1000)
	printTest(LG,['FFT'],True,True,1000)
	return

def printTest(classifier,features,normalize,PCA,PCA_components):
	bdata = loadValidationFS(features,normalize,PCA,PCA_components)
	print "FEATURES: - "+str(features)
	print "ACCURACY FROM "+str(classifier)
	foldsResult = kfoldsTest(classifier,10,bdata)
	a = map(keyToGenre,foldsResult[0]['truth'],)
	b = map(keyToGenre,foldsResult[0]['predictions'])
	print foldsResult[1]
	plot_confusion_matrix(confusion_matrix(a,b),genres().keys(),normalize=True)
	plt.show()
	print ''

#############Functions for validating############

#k fold validation test
def kfoldsTest(classifier,k,data):
	differenceResult = {}
	differenceResult["predictions"] = []
	differenceResult["truth"] = []
	normalize = True
	pca = True
	datalength = len(data)
	indexList = range(0,datalength)
	foldSize = datalength/k
	shuffle(indexList)
	chunks = [indexList[x:x+foldSize] for x in xrange(0, datalength, foldSize)]
	results = []
	for chunk in chunks:
		listCopy = copy.copy(chunks)
		listCopy.remove(chunk)
		trainIndex = reduce(lambda x,y: x+y, listCopy)
		testIndex = chunk
		data['prediction'] = 0
		train = data.loc[trainIndex]
		test = data.loc[testIndex]
		test = classifier(train,test)
		predictions = test['prediction']
		truth = test['Class']
		results.append(calculateAccuracy(test))
		differenceResult["predictions"] = differenceResult["predictions"] + list(predictions)
		differenceResult["truth"] = differenceResult["truth"] + list(truth)
	return [differenceResult,sum(results)/len(results)]

#read dataframe of test set with predictions
#return accuracy
def calculateAccuracy(data):
		correct = float(len(data[data.prediction == data.Class]))
		wrong = float(len(data[data.prediction != data.Class]))
		return correct / (wrong + correct)

#plot a confusion matrix for a set of predictions and truth values
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

#####################Start####################
program()
