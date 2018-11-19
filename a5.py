import re
import numpy as np
from collections import defaultdict
import math

# load up all of the 19997 documents in the corpus

training = sc.textFile("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt")

testing = sc.textFile("s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt")

small_training = sc.textFile("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# gets list of all documents
allDocs = validLines.map(lambda x : x[x.index('id="') + 4 : x.index('" url=')])

# to [("word", (doc, count)),...]
allWordDocOccurrences = None

num_docs = -1

allDictWords = None

allDictWordsSet = None

topWords = None

sortedWordDocOccurrences = None

IDF = None # we'll assign this later

# OTHER POSSIBLE RESOURCES FOR THIS ASSIGNMENT:
# Below is another list of LLH functions
# https://spark.apache.org/docs/latest/mllib-linear-methods.html

mean =  # TODO: calculate the mean and variance of the training data
variance =

x = # feature vectors
y = # takes the form 1 for yes, -1 for no

def lab5():
	global allDictWords, topWords, allDictWordsSet
	regex = re.compile('[^a-zA-Z]')
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
	allCounts = allWords.reduceByKey (lambda a, b: a + b)
	topWords = allCounts.top (20000, lambda x : (x[1],x[0])) # this is altered so that we get the words in the order that we want
	# topWords = allCounts.top (20000, lambda x : 0 - x[1] ) # this is altered so that we get the words in the order that we want
	twentyK = sc.parallelize(range(20000))
	dictionary = twentyK.map (lambda num: (topWords[num][0] , num))
	allDictWords = dictionary.map(lambda x: x[0])
	allDictWordsSet = set(allDictWords.collect())
	dictionary.top(10)
	return dictionary


def consNumpyArray(listPositions, sizeArray):
	#listPositions is a list of (doc, rank), with rank between
	# 1 and sizeArray --> we have to subtract 1 to indx
	countArr = np.zeros(sizeArray)
	for pos in listPositions:
		countArr[pos[1]] += 1
	return countArr


def task1b():
	global corpus, validLines, keyAndText, dictionary, allDocs, num_docs, numWordsInDoc,allWordDocOccurrences, topWords, allDictWords,titles, allDictWordsSet, lenDictionary
	regex = re.compile('[^a-zA-Z]')
	docWordPairList = keyAndText.flatMap(lambda x : ((word ,str(x[0])) for word in regex.sub(' ', x[1]).lower().split()))
	# (word, (docID, posInDictionary)
	wordDocPosRDD = docWordPairList.join(dictionary)
	docPosRDD = wordDocPosRDD.map(lambda x: x[1])
	# ((docID, [(docID, posInDictionary),...]))
	docCollectPosRDD = docPosRDD.groupBy(lambda x: x[0])
	# (doc, [ posInDictionary,...]
	docNumpyWordCounts = docCollectPosRDD.map(lambda x: (x[0], consNumpyArray(x[1],lenDictionary)))
	# results = docNumpyWordCounts.filter(lambda x: x[0] in titles).collect()
	# print(results[0])
	# print(results[1])
	# print(results[2])
	return docNumpyWordCounts

def task1():
	# Build dictionary with 20,000 most frequent words, in decreasing order

	# Find a way to construct our label vector as well


def x_r_calc(x,r):
	return np.dot(x,r)

def llh(x,y,r):
	"""
	
	:param x: 
	:param y: 
	:param r: 
	:return: 
	"""
	n = y.size
	# this first part encompasses the log(1) and the product y*x*r
	x_r = x_r_calc(x,r)
	y_x_r = np.multiply(y,x_r)
	tot_sum = n * math.log(1) + np.sum(y_x_r)

	# second part encompasses term
	# -log(1+e^{x_i*r})
	inner_exp = np.exp(x_r)
	inner_exp += np.full((n,1), 1) # TODO!
	log_term = np.log(inner_exp)
	tot_sum -= log_term.sum()

	# Note: we're using regularaization, so we add the L2 Norm to our Loss Function (LLH)
	# last part encompasses the L2 Norm
	l2_norm = np.sqrt(np.sum(np.square(r)))
	tot_sum += l2_norm

	return tot_sum

def calc_gradient(x,y,r):
	k = x[0].size # TODO: Check all below
	n = y.size
	# should be of dimension (n,k)
	y_tile = np.tile(y,(1,k))
	y_x = np.multiply(y, x)

	# Now calculate the gradient of the second half :
	# -log(1+e^{x_i*r})
	ones = np.full((n,1),1)
	e_x_r = np.exp(x_r_calc(x,r))
	inner_term = np.add(ones, e_x_r)
	quotient = np.divide(-1. * ones, inner_term)
	chain_product = np.multiply(quotient, np.multiply())

	# calculate the gradient of the L2 Norm
	l2_norm_grad = .5 * (2*r)**(-.5)

	# Now we combine together the three vectors
	combined_partial = y_x + chain_product + l2_norm_grad

	return np.sum(combined_partial,0) # want to sum over all columns

def get_mean_vector(x):
	"""
	To be used on the trianing data to get the mean vector. 
	:param x: the matrix of feature vectors
	:return: 
	"""
	x_sum = np.sum(x, 0) # sum the rows together
	n = x.shape[0]
	x_mean = x_sum / (n * 1.)
	return x_mean

def get_sd_vector(x, x_mean):
	"""
	
	:param x: 
	:param x_mean: 
	:return: 
	"""
	n = x.shape[0]
	x_mean_tiled = np.tile(x_mean, (n,1))
	diff = np.subtract(x, x_mean_tiled)
	square_diff = np.square(diff)
	return np.sqrt(np.sum(square_diff,0) / (n*1.))

def normalize_data(x):
	"""
	Normalizes the data and returns the result. 
	Does so using the equation:
	(x - m) / sd
	Where sd = stand dev
	m = mean
	
	:param x: 
	:return: 
	"""
	mean_vector = get_mean_vector(x)
	sd_vector = get_sd_vector(x,mean_vector)
	n = x.shape[0]
	tiled_mean = np.tile(mean_vector, (n,1))
	tiled_sd = np.tile(sd_vector, (n,1))
	normalized_x = np.divide(np.subtract(x, tiled_mean), tiled_sd)
	return normalized_x

def task2():
	"""
	x: Matrix of feature vectors. Dimension (n,k)
	y: Vector of labels. Dimension (n,1)
	r: Matrix of regression coefficients. Has dimension (k, 1)
	Note: k = 20000 in this case
	:return: 
	"""
	global r_estimates, y, x
	THRESH = 10e-6
	BOLD_DRIVER = 1.
	INCREASE = 1.05
	DECREASE = .5


	# Convert each of the documents to a TF-IDF vector
	# Use grad desc to learn a logistic regression model
	# Use L2 regularization
	# Maybe play with parameter controlling of regularization
	x_norm = normalize_data(x)

	r = # TODO: what do we use for starting coefficients?
	dim_data = r.shape[1] # this is the size of our feature vectors (and # regression coeff)
	r = np.append(r,1) # OPTIONAL - add the intercept feature

	# Compute the LLH of your model
	old_llh = llh(x,y,r)
	new_llh = 0

	while abs(old_llh - new_llh) > THRESH:
		grad = calc_gradient(x,y,r)
		# Run until delta-LLH is very small
		new_llh = np.add(old_llh, BOLD_DRIVER * grad)

		if old_llh < new_llh:
			BOLD_DRIVER *= DECREASE
		else:
			BOLD_DRIVER *= INCREASE

		old_llh = new_llh
	return old_llh

def get_k_largest_coeff_indices(k, r):
	"""
	Returns a list of the indices of the top k elements of the r array. 
	Returns indices in order. 
	:param k: specifies the top k coefficients we want
	:param r: the vector of regression coefficients
	:return: 
	"""
	# Got code from
	# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
	# https://stackoverflow.com/questions/44409084/how-to-zip-two-1d-numpy-array-to-2d-numpy-array
	indices = np.argpartition(r, -1 * k)[-1*k:]

	indices_val_zip = np.column_stack(r[indices], indices)
	list_zip = list(indices_val_zip)
	sorted(list_zip, True)

	return (i[1] for i in list_zip)

def get_k_most_imp_words_ordered(k,r,dictionary):
	"""
	To be used with the results from Task 2. 
	 Will get us those words with the largest regression coefficients, 
	 in order of decreasing size (or importance).
	:param k: 
	:param r: vector of coefficients
	:param dictionary: 
	:return: 
	"""
	# TODO: have to be careful if we end up using an intersect feature
	coefficient_indices = get_k_largest_coeff_indices(k,r)
	top_words = []
	for i in coefficient_indices:
		word = dictionary.query(i) # TODO: is this even a thing you can do?
		top_words.append(word)

def task3():
	# First get all the TF-IDF vectors corresponding to the test data
	# Normalize them
	test_x =
	norm_test_x = normalize_data(test_x)

	predicted_y =
	# Evaluate your model

	# Predict whether each point corresponds to australian cases
	# Compute the F1 score obtained by the classifier

	# Retrieve the text of three false positives, for your writeup section

