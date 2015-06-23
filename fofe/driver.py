import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from fofe import FofeVectorizer as FOFE

input_file = 'sample.txt'
docs = [ line for line in open( input_file, 'rb' ) ]

vectorizer = CountVectorizer( binary = True )

x = vectorizer.fit_transform( docs )
print x.toarray()


doc_words = map( lambda x: x.split(), docs )

fofe = FOFE()

fofe_x_dense = fofe.naive_transform( doc_words, vectorizer.vocabulary_ )
fofe_x = fofe.transform( doc_words, vectorizer.vocabulary_ )

print fofe_x_dense
print fofe_x.toarray()
print "Results are the same:", np.allclose( fofe_x_dense, fofe_x.toarray())
