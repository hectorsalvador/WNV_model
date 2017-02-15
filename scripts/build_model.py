# WNV prediction model for the City of Chicago Department of Public Health
# Hector Salvador Lopez, Aug 2016

'''Trains an array of classifiers to obtain a list of models with different 
metrics. Selects the top scoring model, according to the specified metric,
and obtains predictions for WNV occurrence.'''

import argparse
import csv
import queue
import math
import numpy as np
import os
import pandas as pd
from random import random
import sys
import time
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, \
	GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, \
	recall_score, fbeta_score, roc_auc_score, precision_recall_curve,\
	matthews_corrcoef

###############################################################################
## Model class 															     ##
###############################################################################

class Model(object):
	def __init__(self, clf, params):
		self._clf = clf
		self._params = params
		self._tp_list_of_arrays = []
		self._tn_list_of_arrays = []
		self._fp_list_of_arrays = []
		self._fn_list_of_arrays = []
		self._metrics = {}
		self._avg_metrics = {}

	def same_model(self, b):
		if self.clf == b.clf and self.params == b.params: return True
		else: return False

	def add_rates(self, rates):
		self._tp_list_of_arrays.append(np.ndarray.tolist(rates[0])[0])
		self._tn_list_of_arrays.append(np.ndarray.tolist(rates[1])[0])
		self._fp_list_of_arrays.append(np.ndarray.tolist(rates[2])[0])
		self._fn_list_of_arrays.append(np.ndarray.tolist(rates[3])[0])

	def set_metrics(self):
		# for every period
		for t in range(len(self.tp_list_of_arrays)):
			tp_list_at_time_t = self.tp_list_of_arrays[t]
			tn_list_at_time_t = self.tn_list_of_arrays[t]
			fp_list_at_time_t = self.fp_list_of_arrays[t]
			fn_list_at_time_t = self.fn_list_of_arrays[t]

			# for every threshold
			temp = {}
			for r in range(len(tp_list_at_time_t)):
				tp = tp_list_at_time_t[r]
				tn = tn_list_at_time_t[r]
				fp = fp_list_at_time_t[r]
				fn = fn_list_at_time_t[r]
				metrics = get_metric_dict(tp, tn, fp, fn)
				for key, value in metrics.items():
					if key not in temp: temp[key] = []
					temp[key].append(value)

			for metric, value in temp.items():
				self.add_metric(metric, value)

	def add_metric(self, metric, value):
		if metric not in self._metrics:
			self._metrics[metric] = []
		self._metrics[metric].append(value)

	def average_metrics(self, exp = True):
		'''
		Exponential smoothing might be applied to increase weight of 
		most recent	observations.
		https://en.wikipedia.org/wiki/Exponential_smoothing
		'''
		for metric in self._metrics.keys():
			tot = len(self._metrics[metric])
			alpha = 1 - 1 / tot # smoothing parameter
			temp = 0
			sum_weights = 0
			for i in range(0, tot):
				if exp: 
					weight = (1 - alpha) * math.pow(alpha, tot - i)
				else:
					weight = 1 / tot
				temp = temp + np.asarray(self._metrics[metric][i]) * weight
				sum_weights += weight
			self._avg_metrics[metric] = temp / sum_weights

	def find_highest_metric(self, metric):
		max_val = 0
		ix = 0
		for i in range(len(self.avg_metrics[metric])):
			val = float(self.avg_metrics[metric][i])
			if val > max_val:
				max_val = val
				ix = i
		return max_val, ix

	@property
	def clf(self):
		return self._clf
	
	@property
	def params(self):
		return self._params

	@property
	def metrics(self):
		return self._metrics

	@property
	def avg_metrics(self):
		return self._avg_metrics

	@property
	def tp_list_of_arrays(self):
		return self._tp_list_of_arrays
	
	@property
	def tn_list_of_arrays(self):
		return self._tn_list_of_arrays

	@property
	def fp_list_of_arrays(self):
		return self._fp_list_of_arrays

	@property
	def fn_list_of_arrays(self):
		return self._fn_list_of_arrays

	# Visual representation of a Model object
	def __repr__(self):
		clf = self.clf
		params = self.params
		string = 'Model: {}\nParameters: {}\n'.format(clf, params)
		string += 'Average metrics:\n'
		for metric, value in self.avg_metrics.items():
			string += ' - {}: {}\n'.format(metric, value)
		return string

###############################################################################
## Helper functions														     ##
###############################################################################

def get_train_test(df, time, feats):
	'''
	Splits a pd.dataframe into train and test sets, according to indicated time 
	Takes:
		- df,
		- time,
		- feats,

	Returns:
		- X_train, Y_train, X_test, Y_test
	'''
	# Split train set to observations made before time
	X_train = df[df.CHRON < time]
	Y_train = X_train.WNVP
	
	# Process to fit a model -> turn pd.df to np arrays
	train_averages = X_train.mean()
	X_train = preprocess_data(X_train[feats], train_averages)
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	
	# Split test set to observations done on time i
	X_test = df[df.CHRON == time]
	Y_test = X_test.WNVP

	# Process to fit a model -> turn pd.df to np arrays
	X_test = preprocess_data(X_test[feats], train_averages)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	return X_train, Y_train, X_test, Y_test

def preprocess_data(df, averages):
	process_categorical_traps(df)
	rv = df.fillna(averages)
	scaler = StandardScaler()
	scaler.fit(rv)
	rv = scaler.transform(rv)
	return rv

def process_categorical_traps(X):
	le = LabelEncoder()
	le.fit(X.TRAP)
	X.TRAP = le.transform(X.TRAP) 
	le.fit(X.TYPE)
	X.TYPE = le.transform(X.TYPE) 
	
def model_in_mod_list(model, model_list):
	for m in model_list:
		if model.same_model(m): return m
	model_list.append(model)
	return model

def thresh_rates(y, yscores, thresholds):
	tp_l, tn_l, fp_l, fn_l = [], [], [], []
	for t in thresholds:
		yhat = np.asarray([1 if i >= t else 0 for i in yscores])
		tp, tn, fp, fn = get_rates(y, yhat)
		tp_l.append(tp)
		tn_l.append(tn)
		fp_l.append(fp)
		fn_l.append(fn)
	return np.asmatrix([tp_l, tn_l, fp_l, fn_l])

###############################################################################
## Evaluation functions													     ##
###############################################################################

def get_rates(y, ypred):
	'''
	Takes:
		- y, a list of real observations 
		- ypred, a list of predicted values
	'''
	assert len(y) == len(ypred)
	tp, tn, fp, fn = 0, 0, 0, 0

	for i in range(len(y)):
		if y[i] == 1 and ypred[i] == 1:
			tp += 1
		elif y[i] == 1 and ypred[i] == 0:
			fn += 1		
		elif y[i] == 0 and ypred[i] == 0:
			tn += 1		
		elif y[i] == 0 and ypred[i] == 1:
			fp += 1
	return tp, tn, fp, fn

def get_accuracy(tp, tn, fp, fn):
	return (tp + tn) / (tp + tn + fp + fn)

def get_precision(tp, tn, fp, fn):
	if tp == 0 and fp == 0: return 1
	else: return tp / (tp + fp)

def get_recall(tp, tn, fp, fn):
	if tp == 0 and fn == 0: return 1
	else: return tp / (tp + fn)

def get_fbeta(tp, tn, fp, fn, beta = 2):
	'''
	Weight higher the recall in the WNV problem. We care more about
		not missing real positives.
	'''
	if tp == 0 and fn == 0 and fp == 0: return 1
	else:
		return (1 + math.pow(beta, 2)) * tp / \
			(fp + (1 + math.pow(beta, 2)) * tp + fn * math.pow(beta, 2))

def get_youdens_j(tp, tn, fp, fn):
	if (tp == 0) and (fn == 0):
		if (fp == 0) and (tn == 0): return 1 + 1 - 1
		else: return 1 + tn / (fp + tn) - 1
	else:
		return tp / (tp + fn) + tn / (fp + tn) - 1

def get_mcc(tp, tn, fp, fn):
	if (tp == 0 and fp == 0) or (tp == 0 and fn == 0) or \
		(tn == 0 and fp == 0) or (tn == 0 and fn == 0):
		return 0
	else:
		return (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * \
											(tn + fp) * (tn + fn))

def get_metric_dict(tp, tn, fp, fn):
	acc   = get_accuracy(tp, tn, fp, fn)
	prec  = get_precision(tp, tn, fp, fn)
	rec   = get_recall(tp, tn, fp, fn)
	fbeta = get_fbeta(tp, tn, fp, fn)
	f1    = get_fbeta(tp, tn, fp, fn, 1)
	you_j = get_youdens_j(tp, tn, fp, fn)
	mcc   = get_mcc(tp, tn, fp, fn)
	return {'acc': acc, 'fbeta': fbeta, 'f1': f1, 'prec': prec, 'rec': rec,
			'you_j': you_j, 'mcc': mcc}

###############################################################################
## Model class helper functions 										     ##
###############################################################################

def set_average_metrics(model_list):
	for m in model_list:
		m.set_metrics()
		m.average_metrics()

def select_best_k_models(model_list, k, d_metric):
	'''
	Takes:
		- model_list, a dictionary with all models evaluated and their 
			corresponding evaluation metric
		- k, number of best models to retrieve
			d_metric, a string indicating a metric by which models are selected 

	Returns:
		best_k_models, a priority queue with the k models with highest
			d_metric value
	'''
	best_k_models = queue.PriorityQueue(k)
	for model in model_list:
		# since several models have same metric, a random number is added 
		# to the tuples that go in the queue. Otherwise, the heap returns
		# an Inorderable Type error.
		i = random() 
		if best_k_models.full():
			temp = best_k_models.get() # tuple like (max_val, i, ix, model)
			max_val, ix = model.find_highest_metric(d_metric)
			if temp[0] >= max_val: 
				best_k_models.put(temp)
			else:
				best_k_models.put((max_val, i, ix, model))	
		else:
			max_val, ix = model.find_highest_metric(d_metric)
			best_k_models.put((max_val, i, ix, model))
	return best_k_models

def print_top_models(top_models, r):
	rv = []
	while not top_models.empty():
		score, _, i, model = top_models.get()
		print("Highest metric: {}\nThreshold: {}\nModel: {}\n".format(\
		score, r[i], model))
		rv.append((score, _, i, model))
	return rv

###############################################################################
## Classification functions												     ##
###############################################################################

classifiers = { 'LR' : LogisticRegression(),
				'KNN': KNeighborsClassifier(),
				'DT' : DecisionTreeClassifier(),
				'SVM': LinearSVC(),
				'RF' : RandomForestClassifier(),
				'GB' : GradientBoostingClassifier()}

grid = {'LR' : {'penalty': ['l1', 'l2'], 								\
				'C': [0.0001, 0.001, 0.01, 0.05, 0.1, 1, 5, 10, 20],	\
				'n_jobs': [2]}, 
		'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 				\
				'weights': ['uniform', 'distance'], 					\
				'algorithm': ['auto', 'ball_tree', 'kd_tree'],			\
				'n_jobs': [2]},
		'DT' : {'criterion': ['gini', 'entropy'], 						\
				'max_depth': [1, 5, 10, 20],							\
				'max_features': ['sqrt', 'log2'], 						\
				'min_samples_split': [2, 5, 10]},
		'SVM': {'C' : [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10]},
		'RF' : {'n_estimators': [1, 5, 10],								\
				'max_depth': [1, 3, 5, 10],								\
				'max_features': ['sqrt', 'log2'], 						\
				'min_samples_split': [2, 5, 10]},
		'GB' : {'n_estimators': [1, 2, 5, 10, 15], 						\
				'learning_rate' : [0.1, 0.25, 0.5, 0.75, 1],			\
				'subsample' : [0.5, 1.0], 								\
				'max_depth': [1, 3, 5, 8]}}

def classify(X_train, Y_train, X_test, Y_test, clfs, iters, model_list,\
			 r, verbose = False):
	'''
	Takes:
		- X_train, a dataframe of features 
		- Y_train, a dataframe of the label
		- X_test, a dataframe of features 
		- Y_test, a dataframe of the label
		- clfs, a list of strings indicating classifiers to run (e.g. ['LR', 'DT'])
		- iters, an integer referring to iterations of a model
		- model_list, a list with Model objects
		- r, a float to determine if a classifier score yields a positive
			or negative prediction

	Returns:
		Nothing, modifies model_list on site by adding metric results from every
			iteration
	'''
	rates = []

	# for every classifier, try any possible combination of parameters on grid
	for index, clf in enumerate([classifiers[x] for x in clfs]):
		name = clfs[index]
		if verbose: print(name)
		parameter_values = grid[name]
		
		# run the model with all combinations of the above parameters
		for p in ParameterGrid(parameter_values):
			clf.set_params(**p)

			# (Re)Initialize tp/tn/fp/fn list
			rates = []

			# Initialize model, check if it exists in model_list, use it if so
			model = Model(name, p)
			model = model_in_mod_list(model, model_list)

			# run 'iters' number of iterations
			kf_train = KFold(len(X_train), n_folds = iters)
			kf_test  = KFold(len(X_test), n_folds = iters)
			for train_index, _ in kf_train: 
				xtrain, ytrain = X_train[train_index], Y_train[train_index]

				for test_index, _ in kf_test:
					xtest, ytest   = X_test[test_index], Y_test[test_index]
					
					try:
						start_time = time.time()
						if hasattr(clf, 'predict_proba'):
							yscores = clf.fit(xtrain,ytrain).predict_proba(xtest)[:,1]
						else:
							yscores = clf.fit(xtrain,ytrain).decision_function(xtest)
						end_time = time.time()

						# Get tp/tn/fp/fn varying prediction threshold
						rates.append(thresh_rates(ytest, yscores, r))
						
					except IndexError:
						continue

			# Store average tp/tn/fp/fn rates of model p
			temp = rates[0]
			for i in range(1, len(rates)):
				temp = temp + rates[i]
			temp = temp / len(rates)
			model.add_rates(temp)

		if verbose: print('Finished running {}'.format(name))

def loop_by_week(df, p, clfs, iters, r):
	'''
	Goes through every week on the dataset and applies 

	Takes:
		- df, a pandas dataframe with input data for the model
		- p, train model every p weeks
		- clfs, a list with strings referring to classifiers (e.g. ['LR', 'KNN'])
		- iters, number of folds
		- r, a list of floats corresponding to different threshold levels 
			for getting predictions

	Returns:
		- model_list, a list with Model objects

	'''
	FIRST_DAY_OF_2008 = 21
	model_list = []
	cron = list(set(df.CHRON.values))
	cron.sort()
	feats = [col for col in df.columns if col not in ['WNVP', 'CHRON']]

	for i in range(FIRST_DAY_OF_2008, len(cron), p): 
		print('---'*20 + "\nUsing week: {}\n".format(cron[i]) + '---'*20)

		X_train, Y_train, X_test, Y_test = get_train_test(df, cron[i], feats)

		print("Training models...")
		if len(set(np.ndarray.tolist(Y_train))) == 1:
			print("Only one class in train data.")
		elif len(set(np.ndarray.tolist(Y_test))) == 1:
			print("Only one class in test data.")

		classify(X_train, Y_train, X_test, Y_test, clfs, iters, model_list, r)

	set_average_metrics(model_list)

	return model_list

def apply_top_model(top_models_l, df_train, df_pred, r_list):
	'''
	Applies top model accordin to selected metric to prediction data.

	Takes:
		- top_models_l, a list with tuples (e.g. (max_val, rand_int, ix, model))
		- df_train, a pd.dataframe with training data
		- df_pred, a pd.dataframe with data to get predictions

	Saves predictions to 'predictions.csv'
	'''
	# Get best model parameters
	best_model = top_models_l[-1][3]
	r 	  = r_list[top_models_l[-1][2]]
	clf   = best_model.clf
	p 	  = best_model.params
	model = classifiers[clf].set_params(**p)

	# Preprocess training and predictiondata
	feats = [col for col in df_train.columns if col not in ['WNVP', 'CHRON']]
	X_train = df_train
	Y_train = X_train.WNVP
	train_averages = X_train.mean()
	X_train = preprocess_data(X_train[feats], train_averages)
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)

	X_pred = df_pred
	X_pred = preprocess_data(X_pred[feats], train_averages)
	X_pred = np.array(X_pred)

	# Get predictions
	print("Using a {} with {} and r = {}.".format(clf, p, r))
	yscores = model.fit(X_train,Y_train).predict_proba(X_pred)[:,1]
	print(yscores)
	yhat = np.asarray([1 if i >= r else 0 for i in yscores])

	# Dump into a csv
	df_pred['yscores'] = yscores
	df_pred['yhat'] = yhat
	df_pred.to_csv('data/predictions.csv')
	print('Predictions saved in data/predictions.csv.')

###############################################################################
## Wrapper																	 ##
###############################################################################

if __name__ == "__main__":  
	parser = argparse.ArgumentParser(description = \
		'Train models to apply to WNV data.')
	parser.add_argument('--train_i', required = False, 
		default = os.path.dirname(os.path.abspath(sys.argv[0])) + \
		'/../data/train_input.csv',
 		help = 'File with input mosquito and weather data.')
	parser.add_argument('--pred_i', required = False, 
		default = os.path.dirname(os.path.abspath(sys.argv[0])) + \
		'/../data/pred_input.csv',
 		help = 'File with input mosquito and weather data.')
	parser.add_argument('--p', 
		required = False, 
		default = 40, 
		help = 'Train every p periods.')
	parser.add_argument('--iters', 
		required = False, 
		default = 2, 
 		help = 'Number of iterations of every model.')
	parser.add_argument('--k', 
		required = False, 
		default = 5, 
 		help = 'Number of best k models to retrieve.')
	parser.add_argument('--metric', 
		required = False, 
		default = 'fbeta', 
 		help = 'Metric used to select best models.')
	args = parser.parse_args()

	df_train = args.train_i
	periodicity = int(args.p)
	clfs = ['LR', 'KNN', 'DT', 'SVM', 'RF', 'GB']
	iters = args.iters
	r = [i / 100 for i in range(0, 101, 1)]
	d_metric = args.metric
	k = args.k

	df_train = pd.read_csv(df_train)
	model_list   = loop_by_week(df_train, periodicity, clfs, iters, r)
	top_models_q = select_best_k_models(model_list, k, d_metric)
	top_models_l = print_top_models(top_models_q, r)
	print('Analysis done using:')
	print(' - Every {} periods.'.format(periodicity))
	print(' - {} iterations'.format(math.pow(iters, 2)))
	print(' - Retrieving {} models'.format(k))
	print(' - Threshold for prediction scores: {}'.format(r))

	df_preds = args.pred_i
	df_preds = pd.read_csv(df_preds)
	apply_top_model(top_models_l, df_train, df_preds, r)

	print('Done')
