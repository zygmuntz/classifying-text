# classifying-text

Classifying text with bag-of-words, using data from a Kaggle competition: [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/data). Improved version of the original Kaggle tutorial.

	bow_predict.py - train and predict, save a submission file
	bow_validate.py - create train/test split, train, get validation score
	bow_validate_tfidf.py - an improved validation script, with TF-IDF and n-grams
	
	fofe - a directory containing FOFE vectorizer and sample code
	fofe_validate.py - validation scores for count vectorizer vs FOFE
	
	KaggleWord2VecUtility.py - il scripto originale di Kaggle tutoriale
	
See [http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/](http://fastml.com/classifying-text-with-bag-of-words-a-tutorial/) for description.	

## FOFE

Fixed-size Ordinally-Forgetting Encoding is an order-weighted bag-of-words, proposed in _A Fixed-Size Encoding Method for Variable-Length Sequences with its Application to Neural Network Language Models_ ([http://arxiv.org/abs/1505.01504](http://arxiv.org/abs/1505.01504)).

The authors use it with neural networks, but since it's a variation on BoW (and as such it's high-dimensional and sparse), I use it with a linear model. In validation it's slightly better than a vanilla count vectorizer, but worse than TF-IDF. Also, FOFE is sensitive to its one hyperparam, _alpha_.

`fofe/fofe.py` contains a readable, but slow and memory-hungry implementation (`naive_transform`), as well as more efficient function that constructs a sparse matrix (`transform`).

Both these functions expect two arguments: _docs_ and _vocabulary_:
* _docs_ is a list of documents, where each document is a list of words (tokens)
* _vocabulary_ is a dictionary mapping words to indices

You can get a dictionary from `CountVectorizer` - see `fofe_validate.py`.
