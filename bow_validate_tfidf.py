#!/usr/bin/env python

# improved BOW validation script
# changes: leave stopwords in, use TF-IDF vectorizer, removed converting vectorizer output to np.array

import os
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC

from KaggleWord2VecUtility import KaggleWord2VecUtility

#

data_file = 'data/labeledTrainData.tsv'
data = pd.read_csv( data_file, header = 0, delimiter= "\t", quoting = 3 )

train_i, test_i = train_test_split( np.arange( len( data )), train_size = 0.8, random_state = 44 )

train = data.ix[train_i]
test = data.ix[test_i]

#

print "Parsing train reviews..."

clean_train_reviews = []
for review in train['review']:
	clean_train_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

print "Parsing test reviews..."

clean_test_reviews = []
for review in test['review']:
	clean_test_reviews.append( " ".join( KaggleWord2VecUtility.review_to_wordlist( review )))

#

print "Vectorizing..."

vectorizer = TfidfVectorizer( max_features = 40000, ngram_range = ( 1, 3 ), 
	sublinear_tf = True )

train_data_features = vectorizer.fit_transform( clean_train_reviews )
test_data_features = vectorizer.transform( clean_test_reviews )

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



