import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show

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

def model():
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
	model.add(keras.layers.Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	
	return model

in_11, out_11, in_12, out_12, in_13, out_13, in_14, out_14, in_15, out_15, in_all, out_all = data()
print('Data loaded')

in_train = np.concatenate((in_11, in_12))
out_train = np.concatenate((out_11, out_12))
in_val = in_13
out_val = out_13
in_test = np.concatenate((in_14, in_15))
out_test = np.concatenate((out_14, out_15))

model = model()
model.summary()
history = model.fit(in_train, out_train, epochs = 1, validation_data=(in_val, out_val))
test_loss, test_acc = model.evaluate(in_test, out_test, verbose=1)

#plot
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.plot(history.history['acc'], label = 'acc')
plt.plot(history.history['val_acc'], label = 'val_acc')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylabel('acc')
plt.ylim([0.00, 1.00])
plt.legend(loc='lower right')
plt.show() 