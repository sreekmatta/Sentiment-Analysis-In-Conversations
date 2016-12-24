# All imports
from __future__ import division
import csv
import sklearn.svm as svm
import numpy as np
from sklearn.metrics import f1_score
from src import naiveBayesModel
import re

def getWords(sentence):
    words = re.compile('\w+').findall(sentence)
    return words

def extractSentencesAndLabelsForTraining():
    return extractSentencesAndLabels('S:/AI_Projects/SentimentAnalysisInConversation/datasets/trainingDataSetForPredictor.csv')

def extractSentencesAndLabelsForTest():
    return extractSentencesAndLabels('S:/AI_Projects/SentimentAnalysisInConversation/datasets/testDataSetForPredictor.csv')
    
def extractSentencesAndLabels(filename):
    file = open(filename)
    csvFile = csv.reader(file)

    sentences = []
    labels = []
    for row in csvFile:
        sentences.append(row[0])
        labels.append(row[1])
        
    return sentences,labels

def probOfSentiWords(allWordsInCurrSentence,sentiWordsWithProbs):
    np = 0
    nn = 0
    posStrength = 0
    negStrength = 0
    
    for word in allWordsInCurrSentence:
        word = word.lower()
        if sentiWordsWithProbs.get(word) != None:
            probs = sentiWordsWithProbs.get(word)        
            #print probs    
            if probs[2]>probs[3]: # it's a positive word
                np+=1
                posStrength = posStrength + probs[2]
            else:
                nn+=1
                negStrength = negStrength + probs[3]

    probOfnp = 0 
    probOfnn = 0 
    if np+nn !=0:
        probOfnp = np/(np+nn)
        probOfnn = 1-probOfnp
    
    weightedPosStrength = 0
    weightedNegStrength = 0
    
    if np!=0:
        weightedPosStrength = posStrength/np
    if nn!=0:
        weightedNegStrength = negStrength/nn
        
    return  probOfnp, probOfnn,weightedPosStrength,weightedNegStrength

def makeFeatureVector(sentences, labels):
    " construct feature vector" 
    featureVector = []
    
    naiveBayes = naiveBayesModel.NaiveBayesModel()
    naiveBayes.extractSentiWords()
    sentiWordsWithProbs = naiveBayes.getSentiWordsDict()

    for i in range(len(sentences)):
        s = sentences[i]
        allWordsInSentence = getWords(s)
        feature = []
        
        probOfPos,probOfNeg,strengthOfPosivity,strengthOfNegativity = probOfSentiWords(s,sentiWordsWithProbs)
        feature.append(probOfPos)
        feature.append(probOfNeg)
        feature.append(strengthOfPosivity)
        feature.append(strengthOfNegativity)

        feature.append(int(labels[i]))
        featureVector.append(feature)

    return featureVector


def make_np_array_XY(xy):
    """ takes XY (feature + lable) lists, then makes np array for X, Y """
    a = np.array(xy)
    x = a[:,0:-1]
    y = a[:,-1]
    return x,y

def get_f1_score(Y_test, Y_predict):
    test_size = len(Y_test)
    score = 0
    for i in range(test_size):
        if Y_predict[i] == Y_test[i]:
            score += 1
    print 'Got %s out of %s' %(score, test_size) 
    print 'f1 macro = %.2f' %(f1_score(Y_test, Y_predict, average='macro'))
    print 'f1 micro = %.2f' %(f1_score(Y_test, Y_predict, average='micro'))
    print 'f1 weighted = %.2f' %(f1_score(Y_test, Y_predict, average='weighted')) 
    
    print "Accuracy of Support Vector Machine Classifier for the test set with",test_size," size is",100*(f1_score(Y_test, Y_predict, average='weighted'))," %"
                                                                                                           
if __name__ == "__main__":
    
    print "Start of SVM"
    sentencesInTrainSet, labelsInTrainSet = extractSentencesAndLabelsForTraining()
    print "Extraction of Sentences and labels - from training data - done"
    featuresAndLabelsOfTrainSet= makeFeatureVector(sentencesInTrainSet, labelsInTrainSet)        
    print "Creation of Feature Vector is done for training set"
    
    sentencesInTestSet, labelsInTestSet = extractSentencesAndLabelsForTest()
    print "Extraction of Sentences and labels - from training data - done"
    featuresAndLabelsOfTestSet= makeFeatureVector(sentencesInTestSet, labelsInTestSet)        
    print "Creation of Feature Vector is done for training set"
    
    
    numOfFeatures = len(featuresAndLabelsOfTrainSet[0]) - 1

    XY_train = featuresAndLabelsOfTrainSet
    XY_test = featuresAndLabelsOfTestSet
    
    X_train, Y_train = make_np_array_XY(XY_train)
    X_test, Y_test = make_np_array_XY(XY_test)
    print 'len(X_test) = %s len(Y_test) = %s' %(len(X_test), len(Y_test))

    from sklearn import linear_model
    # train set
    print "linear_model.SGDClassifier()..."
    svc = svm.NuSVC().fit(X_train, Y_train)
    
    print "svc.predict()..."
    Y_predict = svc.predict(X_test)
    

    print 'Y_predict:\n', Y_predict
    print 'Y_test:   \n', Y_test

    # get f1 score
    get_f1_score(Y_test, Y_predict)  