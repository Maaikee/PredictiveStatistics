import pandas as pd
import numpy as np
import sklearn, scipy
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
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
	path_11 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2011.csv'
	path_12 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2012.csv'
	path_13 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2013.csv'
	path_14 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2014.csv'
	path_15 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2015.csv'

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

def new_data():
	path_11 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2011.csv'
	path_12 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2012.csv'
	path_13 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2013.csv'
	path_14 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2014.csv'
	path_15 = 'D:\\Maaike\\Scientific Perspectives on GMT\\Assignment 5\\PredictiveStatistics\\Data\\gt_2015.csv'

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

	for col in data_11:
		new_val_0 = col[4] * col[1]
		new_val_1 = (col[3] * col[2] * col[1]) / col[6]

		col = np.delete(col, [0, 1, 2, 3, 4, 6])
		col = np.insert(col, 0, new_val_1)
		col = np.insert(col, 0, new_val_0)

	for col in data_12:
		new_val_0 = col[4] * col[1]
		new_val_1 = (col[3] * col[2] * col[1]) / col[6]

		col = np.delete(col, [0, 1, 2, 3, 4, 6])
		col = np.insert(col, 0, new_val_1)
		col = np.insert(col, 0, new_val_0)

	for col in data_13:
		new_val_0 = col[4] * col[1]
		new_val_1 = (col[3] * col[2] * col[1]) / col[6]

		col = np.delete(col, [0, 1, 2, 3, 4, 6])
		col = np.insert(col, 0, new_val_1)
		col = np.insert(col, 0, new_val_0)

	for col in data_14:
		new_val_0 = col[4] * col[1]
		new_val_1 = (col[3] * col[2] * col[1]) / col[6]

		col = np.delete(col, [0, 1, 2, 3, 4, 6])
		col = np.insert(col, 0, new_val_1)
		col = np.insert(col, 0, new_val_0)

	for col in data_15:
		new_val_0 = col[4] * col[1]
		new_val_1 = (col[3] * col[2] * col[1]) / col[6]

		col = np.delete(col, [0, 1, 2, 3, 4, 6])
		col = np.insert(col, 0, new_val_1)
		col = np.insert(col, 0, new_val_0)


	in_11 = data_11[:,0:5]
	out_11 = data_11[:,6]
	in_12 = data_12[:,0:5]
	out_12 = data_12[:,6]
	in_13 = data_13[:,0:5]
	out_13 = data_13[:,6]
	in_14 = data_14[:,0:5]
	out_14 = data_14[:,6]
	in_15 = data_15[:,0:5]
	out_15 = data_15[:,6]
	in_all = np.concatenate((in_11, in_12, in_13, in_14, in_15))
	out_all = np.concatenate((out_11, out_12, out_13, out_14, out_15))

	return in_11, out_11, in_12, out_12, in_13, out_13, in_14, out_14, in_15, out_15, in_all, out_all


in_11, out_11, in_12, out_12, in_13, out_13, in_14, out_14, in_15, out_15, in_all, out_all = data()
print('Data loaded')

in_train = np.concatenate((in_11, in_12))
out_train = np.concatenate((out_11, out_12))
in_val = in_13
out_val = out_13
in_test = np.concatenate((in_14, in_15))
out_test = np.concatenate((out_14, out_15))

model = LinearRegression().fit(in_train, out_train)
predictions = model.predict(in_val)

SC = scipy.stats.spearmanr(out_val, predictions.flatten()).correlation
MAE = mean_absolute_error(out_val, predictions, multioutput='uniform_average')
RR = (scipy.stats.linregress(out_val, predictions.flatten()).rvalue)**2

print('baseline stats')
print(f'Spearman Correlation Coefficient: {SC}, Mean Absolute Error: {MAE}, R Squared: {RR}')
print(f'Weights: {model.coef_}')

in_combined = np.concatenate((in_train, in_val))
out_combined = np.concatenate((out_train, out_val))

model = LinearRegression().fit(in_combined, out_combined)
predictions = model.predict(in_test)

SC = scipy.stats.spearmanr(out_test, predictions.flatten()).correlation
MAE = mean_absolute_error(out_test, predictions, multioutput='uniform_average')
RR = (scipy.stats.linregress(out_test, predictions.flatten()).rvalue)**2

print(f'Spearman Correlation Coefficient: {SC}, Mean Absolute Error: {MAE}, R Squared: {RR}')

print(f'Weights: {model.coef_}')

statistic, pvalue = scipy.stats.ttest_rel(out_test, predictions)

print(f'Statistic: {statistic}, pvalue: {pvalue}')

# ============= TEST WITH NEW FEATURES =============== #
print('New features stats:')

in_11, out_11, in_12, out_12, in_13, out_13, in_14, out_14, in_15, out_15, in_all, out_all = new_data()

in_combined = np.concatenate((in_11, in_12, in_13))
out_combined = np.concatenate((out_11, out_12, out_13))
in_test = np.concatenate((in_14, in_15))
out_test = np.concatenate((out_14, out_15))

model = LinearRegression().fit(in_combined, out_combined)
predictions = model.predict(in_test)

SC = scipy.stats.spearmanr(out_test, predictions.flatten()).correlation
MAE = mean_absolute_error(out_test, predictions, multioutput='uniform_average')
RR = (scipy.stats.linregress(out_test, predictions.flatten()).rvalue)**2

print(f'Spearman Correlation Coefficient: {SC}, Mean Absolute Error: {MAE}, R Squared: {RR}')

print(f'Weights: {model.coef_}')

statistic, pvalue = scipy.stats.ttest_rel(out_test, predictions)

print(f'Statistic: {statistic}, pvalue: {pvalue}')