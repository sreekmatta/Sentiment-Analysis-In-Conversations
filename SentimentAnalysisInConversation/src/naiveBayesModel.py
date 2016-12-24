# All imports
import csv
import en
import re
from src.constructFrequencyTable import FrequencyTable

# Naive Baye's Model

'''
This Algorithm predicts the friendliness factor of a person using his/her current conversation 
with the help of Naive Baye's Model.       
       
       
stemming and auto word detection code snippet
    
    print naiveBayes.stemming("gave")
    print naiveBayes.stemming("given")
    print naiveBayes.stemming("giver")
    print naiveBayes.stemming("gives")
    print naiveBayes.stemming("persons")
    
    
    print naiveBayes.giveNearestEmotion("gave")
    print naiveBayes.giveNearestEmotion("given")
    print naiveBayes.giveNearestEmotion("giver")
    print naiveBayes.giveNearestEmotion("gives")
    print naiveBayes.spellingCorrections("pesons")
    

'''


# start of algorithm
class NaiveBayesModel:
    
    sentiWordsWithPorbs = {} # dictionary in format "word":[friendliness_probability,unfriendliness_probability]
    neutralWords = ["the","be","to","of","and","a","in","that","have","I","it","for","not","on","with","he",
                             "you","do","at","this","but","his","by","from","they","we","say","her","she","or","an",
                             "my","one","all","would","there","their","what","so","up","out","if","about","who","get",
                             "go","me","when","make","can","like","time","no","just","him","know","take","people","into",
                             "your","good","some","could","them","see","other","than","then","now","look","only","come",
                             "over","think","also","back","after","use","two","how","our","work","first","well","way","even",
                             "new","want","because","any","these","give","day","most","us","as","will","which","year","its"]
                             # we remove the stop words from the current sentence in the conversation
        
    numberOfSentences = 0
    
    def __init__(self):
        numberOfSentences = 0
       
    def getSentiWordsDict(self):
        return self.sentiWordsWithPorbs
         
    def extractSentiWords(self):
        
        file = open('S:/AI_Projects/SentimentAnalysisInConversation/datasets/sentiwordnet.csv')
        csvFile = csv.reader(file)
        
        for row in csvFile:
            
            if self.numberOfSentences!=0: 
                try:
                    self.constructDictOfProbs(row[0],float(row[7]),float(row[8]),float(row[1]),abs(float(row[2])))
                except Exception as err:
                    break
                    
            self.numberOfSentences+=1
        
        file.close()
        
        
        #print self.sentiWordsWithPorbs.get("hate")
        #print self.sentiWordsWithPorbs.get("hurt")
        #print "The friendliness factor of sentence - I hate you - ",self.getProbsOfFriendAndUnfriend("hate")
        #print "The friendliness factor of sentence - You are such a disgust - ",self.getProbsOfFriendAndUnfriend("disgust")

        
    def constructDictOfProbs(self,word,friendProb,unfriendProb,strengthOfPos,strengthOfNeg):
        word = word.lower()
        self.sentiWordsWithPorbs.update({word:[friendProb,unfriendProb,strengthOfPos,strengthOfNeg]})
      
    def predictFriendliness(self,allWordsInConversation,positiveWordsFrequency,negativeWordsFrequency):
        
        probOfFriendliness = 1
        probOfUnfriendliness = 1
        
        for word in allWordsInConversation:
            word = word.lower()
            
            if self.stemming(word) != "":
                word = self.stemming(word)
            
            if word not in self.neutralWords:
                [wordProbOfFriend,wordProbOfUnfriend] = self.getProbsOfFriendAndUnfriend(word,positiveWordsFrequency,negativeWordsFrequency)
                probOfFriendliness = probOfFriendliness * wordProbOfFriend
                probOfUnfriendliness = probOfUnfriendliness * wordProbOfUnfriend

        
        return probOfFriendliness,probOfUnfriendliness
    
        
    
    def getProbsOfFriendAndUnfriend(self,word,positiveWordsFrequency,negativeWordsFrequency):
        probs = [0.6,0.4]
        frequencyInPosWords = 0
        frequencyInNegWords = 0
        
        if positiveWordsFrequency.get(word) != None:
            frequencyInPosWords = positiveWordsFrequency.get(word)
        
        if negativeWordsFrequency.get(word) != None:
             frequencyInNegWords = negativeWordsFrequency.get(word)
         
        n = frequencyInPosWords +  frequencyInNegWords
        if(n!=0):
            likelyhoodP =  float(frequencyInPosWords)/n
            likelyhoodN =  float(frequencyInNegWords)/n
    
            probs = [likelyhoodP,likelyhoodN]
        
        return probs
     
    def stemming(self,word):
        return en.verb.infinitive(word)
        
    def giveNearestEmotion(self,word):
        if en.is_verb(word):
            return en.verb.is_emotion(word, boolean=False)
        
        if en.is_adverb(word):
            return en.adverb.is_emotion(word, boolean=False)
        
        if en.is_adjective(word):
            return en.adjective.is_emotion(word, boolean=False)
        
        return en.noun.is_emotion(word, boolean = False)
        
        
    def spellingCorrections(self,word):
        listOfNearest = en.spelling.suggest(word)
        return listOfNearest[0]
    
    def predictLabelOfEachConversation(self,positiveWordsFrequency,negativeWordsFrequency):
         
        file = open('S:/AI_Projects/SentimentAnalysisInConversation/datasets/testDataSetForPredictor.csv')
        csvFile = csv.reader(file)
        predictedLabels = []
        labelValueByHuman = []
        for row in csvFile:
            sentenceToBePredicted = row[0]
            
            labelValueByHuman.append(int(row[1]))
            allWordsInCurrSentence = self.getWords(sentenceToBePredicted)
            
            probsWithPosAndNeg = self.predictFriendliness(allWordsInCurrSentence,positiveWordsFrequency,negativeWordsFrequency)
            if probsWithPosAndNeg[0] > probsWithPosAndNeg[1]:
                predictedLabels.append(1)
            else:
                predictedLabels.append(0)
                
        return predictedLabels,labelValueByHuman
            
    def getWords(self,sentence):
        return re.compile('\w+').findall(sentence)
        
    def getAccuracyOfNaiveBayes(self,predictedLabels,labelsByHuman):
        i=0
        correctnessCount = 0
        total = len(labelsByHuman)
        
        for i in range(total):
            if int(labelsByHuman[i]) == int(predictedLabels[i]):
                correctnessCount= correctnessCount +1
        
        print "Accuracy of Naive Baye's for the test set with ",total," replies is:",(float(correctnessCount)/total)*100 ," %"
                 
            
def main():
    naiveBayes = NaiveBayesModel()
    naiveBayes.extractSentiWords()
    
    #Construct a frequency table from the training data set 
    freqTable = FrequencyTable()
    freqTable.createTable()
   
     
    #Obtain positive words frequency from the training data set
    positiveWordsFrequency = freqTable.getPositiveWordsFreqFromTrainingSet()

    #Obtain negative words frequency from the training data set
    negativeWordsFrequency = freqTable.getNegativeWordsFreqFromTrainingSet()
    
    predictedLabels,labelsByHuman = naiveBayes.predictLabelOfEachConversation(positiveWordsFrequency,negativeWordsFrequency)
    
    naiveBayes.getAccuracyOfNaiveBayes(predictedLabels, labelsByHuman)
    
if __name__ == "__main__":
    main()

# end of algorithm