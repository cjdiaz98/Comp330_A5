import re
import numpy as np
from collections import defaultdict
import math

# load up all of the 19997 documents in the corpus

training = "s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt"

small_test = "s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt"

allDictWords = None

topWords = None

sortedWordDocOccurrences = None

# OTHER POSSIBLE RESOURCES FOR THIS ASSIGNMENT:
# Below is another list of LLH functions
# https://spark.apache.org/docs/latest/mllib-linear-methods.html


def get_doc_rdd(filename):
	"""
	
	:param filename: 
	:return: 
	A tuple consisting of the following
		-An RDD containing [(doc, text ), ...]
		-The number of documents total in the given file
	"""
	corpus = sc.textFile(filename)

	# each entry in validLines will be a line from the text file
	validLines = corpus.filter(lambda x: 'id' in x)

	# now we transform it into a bunch of (docID, text) pairs
	keyAndText = validLines.map(lambda x: (
		x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:]))
	keyAndText.cache()

	# gets list of all documents
	allDocs = validLines.map(
		lambda x: x[x.index('id="') + 4: x.index('" url=')]) # TODO: do we use this?
	return keyAndText, keyAndText.count()


def lab5():
	global allDictWords, topWords, allDictWordsSet
	regex = re.compile('[^a-zA-Z]')

	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
	allCounts = allWords.reduceByKey (lambda a, b: a + b)
	topWords = allCounts.top (20000, lambda x : (x[1],x[0]))
	twentyK = sc.parallelize(range(20000))
	dictionary = twentyK.map (lambda num: (topWords[num][0] , num))
	allDictWords = dictionary.map(lambda x: x[0])
	dictionary.top(10)
	return dictionary

def consNumpyArray(listPositions, sizeArray):
	#listPositions is a list of (doc, rank), with rank between
	# 1 and sizeArray --> we have to subtract 1 to indx
	countArr = np.zeros(sizeArray)
	for pos in listPositions:
		countArr[pos[1]] += 1
	return countArr

def get_doc_np_arr(keyAndText):
	"""
	
	:param keyAndText: [(docname, String),...]
	:return: 
	"""
	global corpus, dictionary, num_docs, topWords, allDictWords, lenDictionary
	regex = re.compile('[^a-zA-Z]')
	docWordPairList = keyAndText.flatMap(lambda x : ((word ,str(x[0])) for word in regex.sub(' ', x[1]).lower().split()))
	# (word, (docID, posInDictionary)
	wordDocPosRDD = docWordPairList.join(dictionary)
	docPosRDD = wordDocPosRDD.map(lambda x: x[1])
	# ((docID, [(docID, posInDictionary),...]))
	docCollectPosRDD = docPosRDD.groupBy(lambda x: x[0])
	# (doc, [ posInDictionary,...]
	docNumpyWordCounts = docCollectPosRDD.map(lambda x: (x[0], consNumpyArray(x[1],lenDictionary)))

	return docNumpyWordCounts

def cons_feature_vectors(filename):
	"""
	
	:param filename: 
	:return: 
	A numpy array representing the feature vectors of the data. 
	"""
	# load up all of the 19997 documents in the corpus
	corpus = sc.textFile(filename)

	doc_np_arr = get_doc_rdd()
	# TODO: convert this to a numpy array the same way we do with cons_label
	return

def get_key_and_text():

	# each entry in validLines will be a line from the text file
	validLines = corpus.filter(lambda x: 'id' in x)

	# now we transform it into a bunch of (docID, text) pairs
	keyAndText = validLines.map(lambda x: (
		x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:]))


def cons_label_vector(dict):
	"""
	:param dict - an RDD of the form [(doc name, ---),...]
		constructs our label vector 
	:return: RDD of -1's and 1's
	"""
	regex = re.compile('AU(.)*')
	labels = dictionary.map(lambda x: -1 + 2 * regex.match(x[0]))
	np_arr = np.array(labels.collectaslist())
	# TODO: check what exactly we're returning!
	return

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
	k = x.shape[0] # TODO: Check all below
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

def get_k_most_imp_words_ordered(k,r):
	"""
	To be used with the results from Task 2. 
	 Will get us those words with the largest regression coefficients, 
	 in order of decreasing size (or importance).
	:param k: 
	:param r: vector of coefficients
	:return: 
	"""
	global dictionary
	# TODO: have to be careful if we end up using an intersect feature
	coefficient_indices = get_k_largest_coeff_indices(k,r)
	top_words = []
	rev_dict = dictionary.map(lambda entry: (entry[1], entry[0]))
	for i in coefficient_indices:
		word = rev_dict.lookup(i) # TODO: is this even a thing you can do?
		top_words.append(word)

def task1():
	""""""
	global dictionary
	print("Printing out frequencies for:")
	print("applicant")
	print(dictionary.lookup("applicant"))
	print("and")
	print(dictionary.lookup("and"))
	print("attack")
	print(dictionary.lookup("attack"))
	print("protein")
	print(dictionary.lookup("protein"))
	print("car")
	print(dictionary.lookup("car"))

def task2():
	"""
	x: Matrix of feature vectors. Dimension (n,k)
	y: Vector of labels. Dimension (1,n)
	r: Matrix of regression coefficients. Has dimension (1,k)
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

	r = np.ones(x_norm.shape[0])
	r = np.append(r,1) # OPTIONAL - add the intercept feature
	dim_data = r.shape[1] # this is the size of our feature vectors (and # regression coeff)

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

def task3(test_file_name, r):
	""""""
	global small_test
	# First get all the TF-IDF vectors corresponding to the test data
	# Normalize them
	test_x = cons_feature_vectors(small_test)
	norm_test_x = normalize_data(test_x)

	predicted_y =
	# Evaluate your model
	predict(norm_test_x, predicted_y, r)
	# Predict whether each point corresponds to australian cases
	# Compute the F1 score obtained by the classifier

	# Retrieve the text of three false positives, for your writeup section
	false_pos_text =


	return F1, false_pos_text

def predict(x, y, r):
	"""
	
	:param x: 
	:param y: 
	:param r: 
	:return: 
	"""
	correct = 0
	claimed_positives = 0
	actual_positives = 0
	true_positives = 0
	false_positives = []

	for index in range(len(y)):
		if ((np.dot(x[index], r) > 0) and (y[index] > 0)):
			# true positive
			claimed_positives += 1
			actual_positives += 1
			true_positives += 1
			print('success - true positive')
			correct = correct + 1
		elif ((np.dot(x[index], r) < 0) and (y[index] < 0)):
			# true negative
			print('success - true negative')
			correct = correct + 1
		elif ((np.dot(x[index], r) > 0) and (y[index] < 0)):
			claimed_positives += 1
			# false positive
			print('failure - false positive')
			if len(false_positives) < 3:
				false_positives.append(index)
		else:
			actual_positives += 1
			# ((np.dot(x[index], w) < 0) and (y[index] > 0)):
			# false negative
			print('failure - false negative')
	recall = true_positives * 1. / actual_positives
	precision = true_positives * 1. / claimed_positives
	print("True positives: %d. Actual positives: %d .claimed positives: %d"
		  % (true_positives, actual_positives, claimed_positives))
	# print(true_positives * 1. / claimed_positives)
	print("Precision: %f . Recall: %f" % (precision, recall))
	f1_score = (2 * precision * recall) / (precision + recall)
	print('%d out of %d correct.' % (correct, len(y)))
	print("f1 score: %f" % f1_score)

	# TODO: possibly return tuple containing F1 score,
		# and list of three false-positive texts
	return f1_score,

######################
# HERE WE CONSTRUCT THE FEATURE VECTORS AND THE OUTPUT
corpus = None
# allDocs = None

keyAndText, num_docs = get_doc_rdd(training)
dictionary = lab5()
lenDictionary = dictionary.count
x = ()# feature vectors
y = cons_label_vector(dictionary) # takes the form 1 for yes, -1 for no
mean = get_mean_vector(x) # TODO: calculate the mean and variance of the training data
variance = get_sd_vector(x, mean)


F1 = task3(small_test)
