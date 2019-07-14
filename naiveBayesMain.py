import os
import numpy as np
import sklearn
import sys
import os
import getopt
import naiveBayesFunctions as funcpy
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Default corpus name for test and training files
default_train_corpus = "train_corpus"
default_test_corpus = "test_corpus"

# Taking user defined file name: for both test and train. 
arguments, remain = getopt.getopt(sys.argv[1:], 'n')
argument_length = len(sys.argv)

# If user provide the test or train corpus name then our program will execute 
# for that corpus. If he does not give any name then our pre defined corpuses 
# that is not clean will be executed.
if argument_length > 1:
	file_name = sys.argv[argument_length - 1]
	if file_name == "train_corpus_clean" :
		 train_data = file_name
		 if argument_length > 2 :
		 	test_data = sys.argv[argument_length - 2]
		 else:
			test_data = default_test_corpus
	else:
		test_data = file_name
		if argument_length > 2:
			train_data = sys.argv[argument_length - 2]
		else:
			train_data = default_train_corpus
else:
	train_data = default_train_corpus 
	test_data = default_test_corpus

# Find the current directory
dir_name = os.getcwd()

# Accessing the corpus located in the current folder
# TRAIN_DIR =  os.path.join(os.path.expanduser('~'), 'Desktop', 'CPSC_5310', 'project', 'SourceCode', 'train_data')
TRAIN_DIR =  os.path.join(os.path.expanduser('~'), dir_name, train_data)
# TEST_DIR = os.path.join(os.path.expanduser('~'), 'Desktop', 'CPSC_5310', 'project', 'SourceCode', 'test_data')
TEST_DIR =  os.path.join(os.path.expanduser('~'), dir_name, test_data)

# Making Dictionary with the train data
dictionary = funcpy.funcMakeDictionary(TRAIN_DIR)

# Feature extraction part for both train and test data 
features_matrix, labels = funcpy.funcExtractFeatures(TRAIN_DIR)
test_feature_matrix, test_labels = funcpy.funcExtractFeatures(TEST_DIR)

# Call functions for different naive Bayes classifier
funcpy.funcComputeGaussianNB(features_matrix, labels ,  test_feature_matrix, test_labels)
funcpy.funcComputeMultinomialNB(features_matrix, labels ,  test_feature_matrix, test_labels)
funcpy.funcComputeBernoulliNB(features_matrix, labels ,  test_feature_matrix, test_labels)

