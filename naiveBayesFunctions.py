import os
import numpy as np
import sklearn
import sys
import os
import getopt
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Function : for displaying output.
def showScore(test_labels, predicted_labels):

	print "Accuracy Score: ", accuracy_score(test_labels, predicted_labels)
	#print confusion_matrix(test_labels, predicted_labels)
	print "Precision-score : ", precision_score(test_labels, predicted_labels)
	print "Recall-score : ", recall_score(test_labels, predicted_labels)
	print "f1-score (macro) : ", f1_score(test_labels, predicted_labels, average="macro")
	print "f1-score (micro): ", f1_score(test_labels, predicted_labels, average="weighted")
	print ""
	
# Function : For computing the Gaussian Naive Bayes model
def funcComputeGaussianNB(features_matrix, labels ,  test_feature_matrix, test_labels):
	model = GaussianNB()
	model.fit(features_matrix, labels)
	predicted_labels = model.predict(test_feature_matrix)
	print ""
	print "**************************************"
	print "Gaussian Naive Bayes Model:"
	
	# Calling display output function
	showScore(test_labels, predicted_labels)
	

# Function : For computing the Multinomial Naive Bayes model
def funcComputeMultinomialNB(features_matrix, labels ,  test_feature_matrix, test_labels):
	modelMultinomial = MultinomialNB()
	modelMultinomial.fit(features_matrix, labels)
	predicted_labels_multinomial = modelMultinomial.predict(test_feature_matrix)
	
	print "Multinomial Naive Bayes Model:"
	# Calling display output function
	showScore(test_labels, predicted_labels_multinomial)

# Function : For computing the Bernoulli Naive Bayes model
def funcComputeBernoulliNB(features_matrix, labels ,  test_feature_matrix, test_labels):
	modelBernoulli = BernoulliNB()
	modelBernoulli.fit(features_matrix, labels)
	predicted_labels_bernoulli = modelBernoulli.predict(test_feature_matrix)

	print "Bernoulli Naive Bayes Model:"
	# Calling display output function
	showScore(test_labels, predicted_labels_bernoulli)
	print "**************************************"

''' 
	Function: for making dictionary reads training file from 
	the train_corpus (or train_corpus_clean) folder an then construct
	a dictionary for all words of that particular file.
'''
def funcMakeDictionary(train_data_dir):
    global dictionary
    all_words = []

    # Reading all files from train data directory, is sent from main function.
    files = [os.path.join(train_data_dir,f) for f in os.listdir(train_data_dir)]

     # Running a for loop for every file in the folders
    for text in files:
        with open(text) as m:
            for line in m:
		# split the line and store it in words
                words = line.split()
		# Counting the value of total words
                all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()

    for item in list_to_remove:
	# remove the numerical words
        if item.isalpha() == False:
            del dictionary[item]
	# delete the word of lenght 1. 
        elif len(item) == 1:
            del dictionary[item]

        # Making dictionary for most common 300 words
    dictionary = dictionary.most_common(300)

    # return the dictinary to main function
    return dictionary


'''
Function : for extracting features of data. 
we assign a label for every word in dictionary 
and then produce a frequency matrix, number of 
occurring the word in the train or test corpus.
'''
def funcExtractFeatures(data_dir):
    global dictionary
    # Reading all files from data directory, is sent from main function.
    files = [os.path.join(data_dir,fi) for fi in os.listdir(data_dir)]
    
    # Initialize the features matrix with zero. As we took most frequent 
    # 300 words while making dictionary, that is why we need to maintain
    # one dimensional array of 300 length for storing feature in matrix. 
    features_matrix = np.zeros((len(files),300))

    # Initially we also nullify the train labels
    train_labels = np.zeros(len(files))
    count = 0;
    docID = 0;

    # Running a for loop for every file in the folders
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
	    # for every word after spliting a line from file then we need to label wordID as zero.
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
		
        train_labels[docID] = 0;
        filepathTokens = fil.split('/')
        lastToken = filepathTokens[len(filepathTokens) - 1]
	# For finding the name of files which are starts with a common prefix. Here we used 
	# msgsui-represents message of suicidal
        if lastToken.startswith("msgsui"):
	    # Lable the file as 1.
            train_labels[docID] = 1;
            count = count + 1
        docID = docID + 1
	# Return the created feature matrix and train label to the main function. 
	# we will need this matrxi to generate the output. 
    return features_matrix, train_labels
