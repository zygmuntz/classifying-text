#!/usr/bin/env python

# FOFE representation

import os
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC

from KaggleWord2VecUtility import KaggleWord2VecUtility

from fofe.fofe import FofeVectorizer

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

print "Creating a vocabulary..."

vectorizer = CountVectorizer()
vectorizer.fit( clean_train_reviews )

alpha = 1 - 1e-3
fofe = FofeVectorizer( alpha )

print "Vectorizing train..."

train_data_features = vectorizer.transform( clean_train_reviews )

print "Vectorizing test..."

test_data_features = vectorizer.transform( clean_test_reviews )

print "Vectorizing train (FOFE)..."

train_docs = [ doc.split() for doc in clean_train_reviews ]
fofe_train_data_features = fofe.transform( train_docs, vectorizer.vocabulary_ )

print "Vectorizing test (FOFE)..."

test_docs = [ doc.split() for doc in clean_test_reviews ]
fofe_test_data_features = fofe.transform( test_docs, vectorizer.vocabulary_ )

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
print "logistic regression AUC with count features:", auc

fofe_auc = train_and_eval_auc( lr, fofe_train_data_features, train["sentiment"], \
	fofe_test_data_features, test["sentiment"].values )
print "logistic regression AUC with FOFE features:", fofe_auc

# counts
# AUC: 0.945084435083

# alpha 0.99
# AUC: 0.945493966157

# 0.999
# AUC: 0.948106276794

# 0.9999
# AUC: 0.945429484413
