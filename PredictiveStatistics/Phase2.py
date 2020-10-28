
import pandas as pd
import numpy as np
import sklearn, scipy
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
def normalize(array):
	a_min = []
	a_max = []
	averages = []
	for i in range(array.shape[1]):
		averages.append(np.average(array[:,i]))
		a_min.append(array[:,i].min())
		a_max.append(array[:,i].max())

	for a in array:
		for i in range(len(a)):
			val = a[i]
			a[i] =  (val - averages[i]) / (a_max[i] - a_min[i])

	return array

def data():
	path_11 = "C:\\Users\\kikih\Documents\\Multimedia Retrieval\PredictiveStatistics\\Data\\gt_2011.csv"
	path_12 = 'C:\\Users\\kikih\Documents\\Multimedia Retrieval\PredictiveStatistics\\Data\\gt_2012.csv'
	path_13 = 'C:\\Users\\kikih\Documents\\Multimedia Retrieval\PredictiveStatistics\\Data\\gt_2013.csv'
	path_14 = 'C:\\Users\\kikih\Documents\\Multimedia Retrieval\PredictiveStatistics\\Data\\gt_2014.csv'
	path_15 = 'C:\\Users\\kikih\Documents\\Multimedia Retrieval\PredictiveStatistics\\Data\\gt_2015.csv'

	data_11  = pd.read_csv(path_11).values
	data_12  = pd.read_csv(path_12).values
	data_13  = pd.read_csv(path_13).values
	data_14  = pd.read_csv(path_14).values
	data_15  = pd.read_csv(path_15).values

	data_11 = normalize(data_11)
	data_12 = normalize(data_12)
	data_13 = normalize(data_13)
	data_14 = normalize(data_14)
	data_15 = normalize(data_15)

	in_11 = data_11[:,0:9]
	out_11 = data_11[:,10]
	in_12 = data_12[:,0:9]
	out_12 = data_12[:,10]
	in_13 = data_13[:,0:9]
	out_13 = data_13[:,10]
	in_14 = data_14[:,0:9]
	out_14 = data_14[:,10]
	in_15 = data_15[:,0:9]
	out_15 = data_15[:,10]
	in_all = np.concatenate((in_11, in_12, in_13, in_14, in_15))
	out_all = np.concatenate((out_11, out_12, out_13, out_14, out_15))

	return in_11, out_11, in_12, out_12, in_13, out_13, in_14, out_14, in_15, out_15, in_all, out_all

def combine_columns_div(ndarray, col1, col2):
	teycdp = []
	for V in ndarray[:][col1]:
		for W in ndarray[:][col2]:
			teycdp.append(V / W)
	if col1 > col2:
		lastCol = col1
	else:
		lastCol = col2
	ndarray = np.delete(ndarray, lastCol, 1)
	i = 0
	for T in teycdp:
		ndarray[i, col1] = T
		++i
	return ndarray

def combine_columns_mult(ndarray, col1, col2):
	teycdp = []
	for V in ndarray[:][col1]:
		for W in ndarray[:][col2]:
			teycdp.append(V * W)
	if col1 > col2:
		lastCol = col1
	else:
		lastCol = col2
	ndarray = np.delete(ndarray, lastCol, 1)
	i = 0
	for T in teycdp:
		ndarray[i, col1] = T
		++i
	return ndarray

in_11, out_11, in_12, out_12, in_13, out_13, in_14, out_14, in_15, out_15, in_all, out_all = data()
print('Data loaded')

in_train = np.concatenate((in_11, in_12))
out_train = np.concatenate((out_11, out_12))

in_val = in_13
out_val = out_13

in_test = np.concatenate((in_14, in_15))
out_test = np.concatenate((out_14, out_15))

in_combined = np.concatenate((in_train, in_val))
out_combined = np.concatenate((out_train, out_val))

#in_train = combine_columns_mult(in_train, 7, 3)
#in_train = combine_columns_mult(in_train, 7, 5)
#in_train = combine_columns_div(in_train, 2, 3)

#in_val = combine_columns_mult(in_val, 7, 3)
#in_val = combine_columns_mult(in_val, 7, 5)
#in_val = combine_columns_div(in_val, 2, 3)

model = LinearRegression().fit(in_train, out_train)
predictions = model.predict(in_val)

oriSC = scipy.stats.spearmanr(out_val, predictions.flatten()).correlation
oriMAE = mean_absolute_error(out_val, predictions, multioutput='uniform_average')
oriRR = (scipy.stats.linregress(out_val, predictions.flatten()).rvalue)**2

print(f'Spearman Correlation Coefficient: {oriSC}, Mean Absolute Error: {oriMAE}, R Squared: {oriRR}')
print(f'Weights: {model.coef_}')

bestSC = oriSC
bestSC_i = -1
bestSC_j = -1
bestSC_MAE = oriMAE;
bestSC_RR = oriRR;

bestMAE = oriMAE
bestMAE_i = -1
bestMAE_j = -1
bestMAE_SC = oriSC;
bestMAE_RR = oriRR;

bestRR = oriRR
bestRR_i = -1
bestRR_j = -1
bestRR_MAE = oriMAE;
bestRR_SC = oriSC;


for i in range(len(in_train[0])):
	for j in range(len(in_train[0])):
		#print(f'i: {i}, j: {j}')
		combo_train = combine_columns_div(in_train, i, j)
		combo_val = combine_columns_div(in_val, i, j)
		combo_test = combine_columns_div(in_test, i, j)

		in_combined = np.concatenate((combo_train, combo_val))
		out_combined = np.concatenate((out_train, out_val))

		model = LinearRegression().fit(combo_train, out_train)
		predictions = model.predict(combo_val)

		SC = scipy.stats.spearmanr(out_val, predictions.flatten()).correlation
		MAE = mean_absolute_error(out_val, predictions, multioutput='uniform_average')
		RR = (scipy.stats.linregress(out_val, predictions.flatten()).rvalue)**2

		if RR > bestRR:
			bestRR = RR
			bestRR_i = i
			bestRR_j = j
			bestRR_MAE = MAE
			bestRR_SC = SC
		if MAE < bestMAE:
			bestMAE = MAE
			bestMAE_i = i
			bestMAE_j = j
			bestMAE_RR = RR
			bestMAE_SC = SC
		if SC > bestSC:
			bestSC = SC
			bestSC_i = i
			bestSC_j = j
			bestSC_MAE = MAE
			bestSC_RR = RR

	

if i >= 0:
	print(f'Best combo RR = {bestRR} is ({bestRR_i},{bestRR_j}). MAE = {bestRR_MAE}, SC = {bestRR_SC}')
else:
	print(f'Unable to find an improvement for RR')
if i >= 0:
	print(f'Best combo MAE = {bestMAE} is ({bestMAE_i},{bestMAE_j}). RR = {bestMAE_RR}, SC = {bestMAE_SC}')
else:
	print(f'Unable to find an improvement for MAE')
if i >= 0:
	print(f'Best combo SC = {bestSC} is ({bestSC_i},{bestSC_j}). MAE = {bestSC_MAE}, RR = {bestSC_RR}')
else:
	print(f'Unable to find an improvement for SC')

def hist(in_train):
	j=0
	hist_data = []
	sns.set_theme()
	sns.set_context("paper")
	sns.color_palette('pastel')
	f, axes = plt.subplots(3, int(len(in_train[0])/3+0.5), figsize=(15, 5), sharex=True, sharey=True)
	for C in in_train[:]:
		i=0	
		feature = []
		for R in in_train:		
			feature.append(R[j])			
			i+=1	
	
		hist_data.append(feature)
		j+=1
		if j > len(in_train[0])-1:
			break
	x, c = 0, 0
	while x < 3:
		y=0
		while y < int(len(in_train[0])/3+0.5):
			if c > len(hist_data)-1:
				break
			plt.xlabel('Data')
			sns.histplot(data=hist_data[c], bins=50, stat='probability', element="step", ax=axes[x,y], binrange=[-1,1])
			axes[x,y].set_title(f'histogram for feature #{c}', fontweight='bold')
			axes[x,y].set_xlabel('Value of feature')
			y+=1
			c+=1
		x+=1

	plt.ylabel('Probability')
	plt.suptitle(f'histogram and boxplot of features')
	plt.show()
		

