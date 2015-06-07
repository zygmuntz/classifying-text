#!/usr/bin/env python

# validation version of the kaggle BoW script
# changes: train/test split, added logistic regression for comparison with random forest, run rf 10x

import os
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC

from KaggleWord2VecUtility import KaggleWord2VecUtility

#

data_file = 'data/labeledTrainData.tsv'
data = pd.read_csv( data_file, header = 0, delimiter="\t", quoting = 3 )

train_i, test_i = train_test_split( np.arange( len( data )), train_size = 0.8, random_state = 44 )

train = data.ix[train_i]
test = data.ix[test_i]

# train features
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list

print "Cleaning and parsing the training set movie reviews...\n"

for review in train['review']:
	clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review, True )))


# ****** Create a bag of words from the training set
#
print "Creating the bag of words...\n"

vectorizer = CountVectorizer( analyzer = "word", tokenizer = None, preprocessor = None, 
	stop_words = None, max_features = 5000 )

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# test features

# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"

for review in test['review']:
	clean_test_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review, True )))

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform( clean_test_reviews )
test_data_features = test_data_features.toarray()

###

print "Training the random forest (this may take a while)..."

forest = RandomForestClassifier( n_estimators = 100, n_jobs = -1, verbose = 1 )
forest = forest.fit( train_data_features, train["sentiment"] )

print "Predicting test labels...\n"
rf_p = forest.predict_proba( test_data_features )

auc = AUC( test['sentiment'].values, rf_p[:,1] )
print "random forest AUC:", auc

# a random score from a _random_ forest
# AUC: 0.919056767104


# let's define a helper function

def train_and_eval_auc( model, train_x, train_y, test_x, test_y ):
	model.fit( train_x, train_y )
	p = model.predict_proba( test_x )
	auc = AUC( test_y, p[:,1] )
	return auc

#

lr = LR()
auc = train_and_eval_auc( lr, train_data_features, train["sentiment"], \
	test_data_features, test["sentiment"].values )
print "logistic regression AUC:", auc

# logistic regression AUC: 0.925748792247
# logistic regression AUC: 0.928301070895	# different split


# train a random forest ten times, average the scores

rf_aucs = []
for i in range( 10 ):
	auc = train_and_eval_auc( forest, train_data_features, train["sentiment"], \
		test_data_features, test["sentiment"].values )
	
	print "random forest run {}, AUC: {}".format( i, auc )
	rf_aucs.append( auc )
	
avg_auc = sum( rf_aucs ) / len( rf_aucs )
print "Average AUC from random forest:", avg_auc





