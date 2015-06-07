#!/usr/bin/env python

# train and predict, based on validation params

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression as LR

from KaggleWord2VecUtility import KaggleWord2VecUtility

#

train_file = 'data/labeledTrainData.tsv' 
test_file = 'data/testData.tsv'
output_file = 'data/bow_predictions.csv'

#

train = pd.read_csv( train_file, header = 0, delimiter = "\t", quoting = 3 )
test = pd.read_csv( test_file, header = 0, delimiter = "\t", quoting = 3 )

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

print "Vectorizing train..."

vectorizer = TfidfVectorizer( max_features = 40000, ngram_range = ( 1, 3 ), 
	sublinear_tf = True )
train_x = vectorizer.fit_transform( clean_train_reviews )

print "Vectorizing test..."

test_x = vectorizer.transform( clean_test_reviews )

print "Training..."

model = LR()
model.fit( train_x, train["sentiment"] )
p = model.predict_proba( test_x )[:,1] 

#

print "Writing results..."

output = pd.DataFrame( data = { "id": test["id"], "sentiment": p } )
output.to_csv( output_file, index = False, quoting = 3 )



