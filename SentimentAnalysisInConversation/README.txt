Welcome to the project Sentiment Analysis in Conversations
----------------------------------------------------------

Steps to run this project:
--------------------------
1) Firstly open the command prompt from the current folder SentimentAnalysisInConversations

2) Then run the respective files naiveBayesModel.py and SVMClassifier.py

Steps to run Naive Baye's Algorithm on considered test set =>
To get the accuracy of the Naive Baye's model for the given test set, we need to run the
file naiveBayesModel.py as follows :

> python naiveBayesModel.py

Steps to run Support Vector Machine Algorithm on considered test set =>
To get the accuracy of the Support Vector Machine training Algorithm for the given test set, 
we need to run the file SVMClassifier.py as follows :

> python SVMClassifier.py

Filenames of the datasets in the python files:
----------------------------------------------
Please provide the filenames of the respective datasets of your current folder that you have downloaded 
this project to before running the files naiveBayesModel.py and SVMClassifier.py

The file links in the following python files have to be changed:

line 55:  file = open('S:/AI_Projects/SentimentAnalysisInConversations/datasets/trainingDataSetForPredictor.csv')
in constructFreqencyTable.py

line 61:  file = open('S:/AI_Projects/SentimentAnalysisInConversation/datasets/sentiwordnet.csv')
line 150: file = open('S:/AI_Projects/SentimentAnalysisInConversation/datasets/testDataSetForPredictor.csv')
in naiveBayesModel.py

line 17 and 20 in SVMClassifier.py
Data sets:
----------
The datasets (both training and the test data) for this project have been created by us which 
is specifically targeted for this application. Also we are using a sentiwordnet dataset from online
resource. 

All the 3 data sets are located in the folder datasets


Additional Packages:
--------------------
The project has the following dependencies:
NumPy
SciPy
matplotlib
Scikit_learn