import numpy as np
from scipy.sparse import coo_matrix

class FofeVectorizer():

	def __init__( self, alpha = 0.99 ):
		self.alpha = alpha
		
	def naive_transform( self, docs, vocabulary ):
		x = np.zeros(( len( docs ), len( vocabulary )))
		for row_i, doc in enumerate( docs ):
			for word in doc:
				x[row_i,:] *= self.alpha
				try:
					col_i = vocabulary[word]
				except KeyError:
					# not in vocabulary: a one-letter word or a word from a test set
					continue
				x[row_i, col_i] += 1
		return x

		
	def transform( self, docs, vocabulary ):

		# pre-compute alpha powers
		alpha_powers = { x: self.alpha ** x for x in range( 10000 ) }

		data = []
		i = []
		j = []

		for r, doc in enumerate( docs ):

			doc_len = len( doc )
			
			# row indices for the doc
			i += [ r for _ in range( doc_len ) ]
		
			for word_pos, word in enumerate( doc ):
				
				# column index for the word
				try:
					word_i = vocabulary[word]
					j.append( word_i )
				except KeyError:
					# not in vocabulary: a one-letter word or a word from a test set
					i.pop()
					continue

				# value at [i,j]; duplicates will be added up
				try:
					data.append( alpha_powers[ doc_len - word_pos - 1 ] )
				except KeyError:
					data.append( alpha ** ( doc_len - word_pos - 1 ))
			
		"""	
		print data
		print i
		print j
		"""
		
		return coo_matrix(( data, ( i, j )), ( len( docs ), len( vocabulary )))	
		