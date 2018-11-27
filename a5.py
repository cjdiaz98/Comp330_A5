import re
import numpy as np
from collections import defaultdict
import math
import operator as opr

allDictWords = None

topWords = None

sortedWordDocOccurrences = None

# OTHER POSSIBLE RESOURCES FOR THIS ASSIGNMENT:
# Below is another list of LLH functions
# https://spark.apache.org/docs/latest/mllib-linear-methods.html

IDF = None

PENALTY = .01

def get_doc_rdd(filename):
	"""
	:param filename: 
	:return: 
	A tuple consisting of the following
		-An RDD containing [(doc, text ), ...]
		-A map of 
	"""
	corpus = sc.textFile(filename)
	# each entry in validLines will be a line from the text file
	validLines = corpus.filter(lambda x: 'id' in x)
	# now we transform it into a bunch of (docID, text) pairs
	keyAndText = get_key_and_text(filename)
	# gets list of all documents
	# allDocs = validLines.map(
	# 	lambda x: x[x.index('id="') + 4: x.index('" url=')]) # TODO: do we use this?
	numWordsInDoc = keyAndText.map(lambda x: (x[0], len(x[1])))
	return keyAndText, numWordsInDoc


def lab5():
	global allDictWords, topWords, keyAndText
	regex = re.compile('[^a-zA-Z]')
	# keyAndText = get_key_and_text(filename)
	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))
	allCounts = allWords.reduceByKey (lambda a, b: a + b)
	topWords = allCounts.top (20000, lambda x : (x[1],x[0]))
	twentyK = sc.parallelize(range(20000))
	dictionary = twentyK.map (lambda num: (topWords[num][0] , num))
	# allDictWords = dictionary.map(lambda x: x[0])
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
	global dictionary, num_docs, topWords, lenDictionary
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

def cons_training_feature_rdd(filename):
	"""
	
	:param keyAndText: 
	:return: 
	A numpy array representing the feature vectors of the data. 
	This is used to construct the 
	"""
	global IDF
	keyAndText, numWordsInDoc = get_doc_rdd(filename)
	docAndFrequencies = cons_frequency_rdd(keyAndText)
	IDF = cons_IDF_mat(docAndFrequencies)
	tf_idf_rdd = cons_TF_IDF(docAndFrequencies, numWordsInDoc, IDF)
	# tf_idf_np_rdd = tf_idf_rdd.map(lambda x: x[1])
	tf_idf_np_rdd = tf_idf_rdd.map(lambda x: (x[0] ,np.append(x[1],1))) # TODO: this is if we want to add an intercept!
	return tf_idf_np_rdd, keyAndText

def cons_test_feature_rdd(filename):
	"""

	:param keyAndText: 
	:return: 
	A numpy array representing the feature vectors of the data. 
	This is used to construct the 
	"""
	global IDF
	keyAndText, numWordsInDoc = get_doc_rdd(filename)
	docAndFrequencies = cons_frequency_rdd(keyAndText)
	tf_idf_rdd = cons_TF_IDF(docAndFrequencies, numWordsInDoc, IDF)
	# tf_idf_np_rdd = tf_idf_rdd.map(lambda x: x[1])
	tf_idf_np_rdd = tf_idf_rdd.map(lambda x: (x[0] ,np.append(x[1],1))) # TODO: this is if we want to add an intercept!
	tf_idf_np = np.array(tf_idf_np_rdd.collect())  # gets us a list of lists
	return tf_idf_np, keyAndText

def get_key_and_text(filename):
	corpus = sc.textFile(filename)
	# each entry in validLines will be a line from the text file
	validLines = corpus.filter(lambda x: 'id' in x)
	# now we transform it into a bunch of (docID, text) pairs
	keyAndText = validLines.map(lambda x: (
		x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:]))
	keyAndText.cache()
	return keyAndText

def cons_label_rdd(docNameRDD):
	"""
	:param docNameRDD - an RDD of the form [(doc name, ---),...]
		constructs our label vector 
	:return: RDD of -1's and 1's
	Will be of dimension
	"""
	regex = re.compile('AU(.)*')
	labels = docNameRDD.map(lambda x: (x[0], 1. * bool(regex.match(x[0]) )))
	return labels

def x_r_calc(x,r):
	return np.dot(x,r)

def llh(x,y,r):
	"""
	
	:param x: 
	:param y: 
	:param r: 
	:return: 
	"""
	global PENALTY
	n = y.count()
	# this first part encompasses the log(1) and the product y*x*r
	x_r = x.map(lambda z: (z[0], np.sum(np.dot(z[1], r))))
	y_x_r = x_r.join(y).map(lambda z: (1, z[1][0] * z[1][1]) )
	tot_sum = y_x_r.reduceByKey(opr.add).lookup(1)[0]
	# second part encompasses term
	# -log(1+e^{x_i*r})
	# neg_log_term = x_r.map(lambda z: (1, -1 * z[1])) # TODO: temporary fix to math range error
	neg_log_term = x_r.map(lambda z: (1, -1 * math.log(1 + math.exp(z[1]))))
	neg_log_term = neg_log_term.reduceByKey(opr.add)
	tot_sum += neg_log_term.lookup(1)[0]
	# Note: we're using regularaization, so we add the L2 Norm to our Loss Function (LLH)
	# last part encompasses the L2 Norm
	l2_norm = PENALTY * np.sqrt(np.sum(np.square(r)))
	tot_sum += l2_norm
	return tot_sum

def calc_gradient(x,y,r):
	"""
	Note: 
	k = # of features
	n = sample size
	:param x: 
	:param y: 
	:param r: 
	:return: 
	"""
	k = x.take(1)[0][1].shape[0]
	n = y.count()
	# should be of dimension (n,k)
	# calculate gradient of y(x*r)
	y_x = x.join(y)
	y_x = y_x.map(lambda z: (1, z[1][0] * z[1][1]))
	y_x = np.transpose(np.asmatrix(y_x.reduceByKey(np.add).lookup(1)[0]))
	# Now calculate the gradient of the second half :
	# -log(1+e^{x_i*r})
	x_r = x.map(lambda z: (z[0], np.sum(x_r_calc(z[1], r))))
	e_x_r = x_r.map(lambda z: (z[0], math.exp(z[1])))
	denom = e_x_r.map(lambda z: (z[0], -1. / (1 + z[1])))
	combined_log_term = e_x_r.join(denom).join(x)
	combined_log_term = combined_log_term.map(lambda z: (1,z[1][0][0] * z[1][0][1] * z[1][1]))
	combined_log_term = np.transpose(np.asmatrix(combined_log_term.reduceByKey(lambda a,b: np.add(a,b)).lookup(1)))
	# calculate the gradient of the L2 Norm
	# ones_k = np.full((k,1),1)
	# sqrt_r = np.sqrt(np.abs(np.square(r))) # todo: no longer need
	l2_norm_grad = PENALTY * 2 * r # (k,1)
	# l2_norm_grad = PENALTY * .5 * np.divide(ones_k, sqrt_r) # (k,1) # TODO: ond gradient calculation
	# Now we combine together the three vectors
	combined_partial = y_x + combined_log_term + l2_norm_grad
	combined_partial /= x.count() # TODO: maybe get rid of?
	return combined_partial # (k,1)

def get_mean_vector(x):
	"""
	To be used on the trianing data to get the mean vector. 
	:param x: the matrix of feature vectors
	:return: 
	"""
	x_sum = x.map(lambda z: (1, z[1]))
	x_sum = x_sum.reduceByKey(np.add).lookup(1)[0] # sum the rows together
	n = x.count()
	x_mean = x_sum / (n * 1.)
	return x_mean

def get_sd_vector(x, x_mean):
	"""
	
	:param x: 
	:param x_mean: 
	:return: 
	"""
	n = x.count()
	x_diff = x.map(lambda z: (1, np.square(np.subtract(z[1], x_mean))))
	diff_sum = x_diff.reduceByKey(np.add).lookup(1)[0] # sum the rows together
	return np.sqrt(diff_sum / (n-1) * 1.)

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
	sd_vector[sd_vector == 0] = 1
	# todo: needed this ^ because for some reason, I got SD's of 0
	normalized_x = x.map(lambda z: (z[0], np.divide(np.subtract(z[1], mean_vector), sd_vector)))
	return normalized_x

def check_mean_0(x):
	"""
	Returns a list of the indices of the vector x where the 
	mean isn't equal to 0 or really close
	:param x: 
	:return: 
	"""
	thresh = 1e-2
	mean_x = get_mean_vector(x)
	return np.where(np.abs(mean_x) > thresh)

def check_sd_1(x):
	thresh = 1e-2
	mean_x = get_mean_vector(x)
	sd = get_sd_vector(x, mean_x)
	diff = np.abs(np.subtract( np.full(sd.shape,1), sd))
	return np.where(diff > thresh)


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
	abs_r = np.abs(r)
	indices = np.argpartition(abs_r, -1 * k)[-1*k:]

	indices_val_zip = np.column_stack(abs_r[indices], indices)
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

def cons_frequency_rdd(keyAndText):
	"""
	Pretty much a clone of my A4 solution.
	:return: 
	"""
	global dictionary, lenDictionary
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

def cons_IDF_mat(docAndFrequencies):
	wordDocSingleOccurrences = docAndFrequencies.map(
		lambda x: (1, np.clip(x[1], 0, 1)))  # map all to the same key
	wordDocAllOccurrencesRDD = wordDocSingleOccurrences.reduceByKey(
		opr.add)
	allWordDocOccurrencesArr = wordDocAllOccurrencesRDD.lookup(1)
	idf = np.full(lenDictionary, num_docs)
	idf = np.divide(idf, allWordDocOccurrencesArr)
	idf = np.log(idf)
	return idf

def cons_TF_IDF(docAndFrequencies, numWordsInDoc, IDF):
	"""
	Will be called
	Note, that the only things that change call to call are the rdd's corresponding to
	the documents and the frequencies of words, and the number of words per each doc. 
	This is because these things are used to calculate the TF portion of the TFIDF.
	It is only the TF that changes when we're calculating the TF-IDF for training vs 
	test docs. 
	
	:param docAndFrequencies: 
	:param numWordsInDoc: 
	:return: 
	"""
	global num_docs, lenDictionary,allWordDocOccurrences, dictionary, sortedWordDocOccurrences, topWords, titles_of_interest
	pre_TF_d = docAndFrequencies.join(numWordsInDoc)
	TF_d = pre_TF_d.map(lambda x: (x[0], x[1][0] / (x[1][1] * 1.) ) )
	TF_IDF = TF_d.map(lambda x: (x[0], x[1] * IDF))
	return TF_IDF

def task2(x,y,r):
	"""
	x: Matrix of feature vectors. Dimension (n,k)
	y: Vector of labels. Dimension (n,1)
	r: Matrix of regression coefficients. Has dimension (k,1)
	Note: k = 20000 in this case
	:return: 
	"""
	THRESH = 10e-4
	BOLD_DRIVER = 1.
	INCREASE = 1.05
	DECREASE = .5
	# Convert each of the documents to a TF-IDF vector
	# Use grad desc to learn a logistic regression model
	# Use L2 regularization
	# Maybe play with parameter controlling of regularization
	x_norm = normalize_data(x)
	# Compute the LLH of your model
	old_llh = llh(x_norm,y,r)
	new_llh = 0
	while abs(old_llh - new_llh) > THRESH:
		print("LLH %f" % old_llh)
		grad = calc_gradient(x_norm,y,r)
		# Run until delta-LLH is very small
		r = r + BOLD_DRIVER * grad
		new_llh = llh(x_norm,y,r)
		if old_llh > new_llh:
			BOLD_DRIVER *= DECREASE
			# we want to de-incentivize decreases
		else:
			BOLD_DRIVER *= INCREASE
		old_llh = new_llh
	return r

def task3(test_file_name, r):
	""""""

	# TODO: might not even have to do the below code.
		# Only do if you need a way to index in to find a text block
	# rangeN = sc.parallelize(range(testKeyAndText.count()))
	# docAndOrder = testKeyAndText.map(lambda x: ()) # [(0, docName), (1, docName), ...]

	test_x, testKeyAndText = cons_test_feature_rdd(test_file_name)
	norm_test_x = normalize_data(test_x)

	actual_y = cons_label_rdd(testKeyAndText)

	# Evaluate your model
	f1, false_pos_indices = predict(norm_test_x, actual_y, r)
	# Predict whether each point corresponds to australian cases
	# Compute the F1 score obtained by the classifier

	# Retrieve the text of three false positives, for your writeup section
	false_pos_text = []
	for i in false_pos_indices:
		false_pos_text.append(testKeyAndText[i]) # TODO: can we do this???
	return F1, false_pos_text

def predict(x, y, r):
	"""
	Note: we can do this with bulk matrix operations, but then we wouldn't 
	be able to tell what types of positive and negatives tests we got. 
	
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
	x_y = x.join(y)

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
	return f1_score,false_positives

######################
# HERE WE CONSTRUCT THE FEATURE VECTORS AND THE OUTPUT
corpus = None
# allDocs = None

training = "s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt"
small_test = "s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt"
small_training = "s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt"

###### TODO: COMPUTE SMALL  ########
# keyAndText = get_key_and_text(training)
smallKeyAndText = get_key_and_text(small_training)
# keyAndText.cache()
small_num_docs = smallKeyAndText.count()
small_dictionary = lab5()
# dictionary.cache()
smallLenDictionary = small_dictionary.count()

small_x, smallKeyAndText = cons_training_feature_rdd(small_training) # feature vectors
# x, keyAndText = cons_feature_vectors(training) # feature vectors
small_y = cons_label_rdd(smallKeyAndText) # takes the form 1 for yes, 0 for no

##############

###### TODO: COMPUTE BIG ########
bigKeyAndText = get_key_and_text(training)
# keyAndText.cache()
big_num_docs = bigKeyAndText.count()
big_dictionary = lab5()
# dictionary.cache()
bigLenDictionary = big_dictionary.count()

big_x, bigKeyAndText = cons_training_feature_rdd(training) # feature vectors
# x, keyAndText = cons_feature_vectors(training) # feature vectors
big_y = cons_label_rdd(bigKeyAndText) # takes the form 1 for yes, 0 for no
##############

dictionary = small_dictionary
lenDictionary = smallLenDictionary
num_docs = small_num_docs
x = small_x
y = small_y

# NOTE: make sure to assign dictionary and lenDictionary to the version that you want

# mean = get_mean_vector(x) # TODO: calculate the mean and variance of the training data
# variance = get_sd_vector(x, mean)

task1()

k = x.take(1)[0][1].size
# initial_r = np.full(k, .1)  # (k,1)
# initial_r = np.asmatrix(np.full((k, 1), .1)) # (k,1)
initial_r = np.asmatrix(np.full((k, 1), .1)) # (k,1)
new_r = task2(x,y,initial_r)

# TODO: WE HAVE THE FOLLOWING TO TEST OUR CODE:
x_norm = normalize_data(x)
old_llh = llh(x_norm,y,initial_r)
grad = calc_gradient(x_norm,y,initial_r)

##################

testKeyAndText = get_key_and_text(small_test)
testKeyAndText, testNumWordsInDoc = get_doc_rdd(small_test) # have this for testing purposes

F1 = task3(small_test, new_r)
