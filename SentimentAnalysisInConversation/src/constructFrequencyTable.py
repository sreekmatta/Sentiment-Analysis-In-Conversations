# All imports
import csv
import re
import en

# Code to construct frequency table

''' We have numerous  datasets.
    all datasets have the predicted replies to the patient with friendliness labeled 
    to each as 0(unfriendly reply) and 1(positive or friendly reply) 
    Now this Algorithm extracts all the words from the sentences to count their occurrences 
    in friendly and unfriendly replies and constructs a frequency table as follows:
    
    Note: This algo do not consider neutral words such as 'This' 'The' 'a' 'an' 'are' 'I' 'You'
    
    For example: -- sample dataset of replies --
                 Reply1: This is a pleasant day.                  label : 1
                 Reply2: I had a pleasant day at your house.      label : 1
                 Reply3: I hate your backyard.                    label : 0
                 Reply4: You are all stupid.                      label : 0  
    
     
    sample dataset of replies    ========> construct FrequncyTable Algorithm =========> Frequency table
    Frequency table :
    +-------------------------------------+
    | words       | friendly | unfriendly |
    +-------------------------------------+                     
    | day         |  2       |  0         |
    | pleasant    |  2       |  0         |
    | house       |  1       |  0         |
    | hate        |  0       |  1         |
    | backyard    |  0       |  1         |
    | stupid      |  0       |  1         |
    | all         |  0       |  1         |
    +-------------+----------+------------+
'''


# start of algorithm
class FrequencyTable:


    def __init__(self):
        self.friendlyWordsFreq = {}
        self.unfriendlyWordsFreq = {}
        self.neutralWords = ["the","be","to","of","and","a","in","that","have","i","it","for","not","on","with","he",
                             "you","do","at","this","but","his","by","from","they","we","say","her","she","or","an",
                             "my","one","all","would","there","their","what","so","up","out","if","about","who","get",
                             "go","me","when","make","can","like","time","no","just","him","know","take","people","into",
                             "your","good","some","could","them","see","other","than","then","now","look","only","come",
                             "over","think","also","back","after","use","two","how","our","work","first","well","way","even",
                             "new","want","because","any","these","give","day","most","us","as","will","which","year","its"]
        
    def createTable(self):
        file = open('S:/AI_Projects/SentimentAnalysisInConversation/datasets/trainingDataSetForPredictor.csv')
        csvFile = csv.reader(file)
        

        for row in csvFile:
            allWords = self.getWords(row[0]) # input  : "Hello, my name is keerthi" , "1"
                                          # output : ['Hello', my', 'name', 'is', 'keerthi'] 
            
            self.constructFrequencyTable(allWords, row[1])
        
        print self.friendlyWordsFreq
        print self.unfriendlyWordsFreq
        
            
    def getWords(self,sentence):
        return re.compile('\w+').findall(sentence)
            
    def stemming(self,word):
        return en.verb.infinitive(word)

    def constructFrequencyTable(self,allWords,labelValue):
        
        for word in allWords:
            word = word.lower()
            
            if self.stemming(word) != "":
                word = self.stemming(word)
                
            if word in self.neutralWords:
                continue
            else:
                if int(labelValue) == 1 :  # positive sentence
                    self.checkIfAlreadyExistsAndAdd(self.friendlyWordsFreq, word)
                else:                 # negative sentence
                    self.checkIfAlreadyExistsAndAdd(self.unfriendlyWordsFreq, word)
                    
    def checkIfAlreadyExistsAndAdd(self,dictData,word):
        if dictData.get(word) == None:
            dictData.update({word:1}) 
        else:
            dictData.update({word:(dictData.get(word)+1)})
     
    def getPositiveWordsFreqFromTrainingSet(self):
        return self.friendlyWordsFreq
    
    def getNegativeWordsFreqFromTrainingSet(self):
        return self.unfriendlyWordsFreq
           
def main():
    freqTable = FrequencyTable()
    freqTable.createTable()

if __name__ == "__main__":
    main()

# end of algorithm