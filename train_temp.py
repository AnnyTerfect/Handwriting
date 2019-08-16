from __future__ import print_function
import keras
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.datasets import mnist
from time import time


def my_func1(x):
	y = K.square(K.relu(x)) - K.square(K.relu(-x))
	y = y / (1 + K.square(x))
	return y

def my_func2(x):
	y = K.square(K.relu(x)) - K.square(K.relu(-x))
	y = y / (1 + K.square(x))
	y = y + x
	return y

def my_func3(x):
	y = x
	y = y - K.square(K.relu(-x)) / (1 + K.square(x))
	return y

def my_func4(x):
	y = 2 * K.square(x - 1) / (1 + K.square(x - 1)) - 1
	y *= K.sign(x)
	return y

def my_func5(x):
	y = K.relu(x)
	y += (-2 * K.square(x - 1) / (1 + K.square(x - 1)) + 1) * K.sign(K.relu(-x))
	return y

def my_func6(x):
	y = K.relu(x) - K.relu(-x) / (1 + K.relu(-x))
	return y

def my_func7(x):
	y = K.relu(x) - K.square(K.relu(-x)) / (1 + K.square(x))
	return y

def my_func8(x):
	y = K.relu(x) - K.sqrt(K.relu(-x)) / (1 + K.sqrt(K.relu(-x)))
	return y

funs = [K.sigmoid, K.softsign, my_func1, my_func2, my_func3, my_func4, my_func5, my_func6, my_func7, my_func8]
#time 5 5 8 9 8 8 10 7 8 9

names = ['sigmoid', 'softsign', 'my_fun1', 'my_fun2', 'my_fun3', 'my_fun4', 'my_fun5', 'my_fun6', 'my_fun7', 'my_fun8']
historys = []
def start():
	for i in range(len(funs)):
		historys.append(get_history(funs[i], epochs=2, lr=0.0001))

def plot(historys, names, item, fig_name):
	plt.figure()
	for i in range(len(historys)):
		plt.plot(historys[i].history[item])
	plt.title(fig_name)
	plt.ylabel(item)
	plt.xlabel('epoch')
	plt.legend(names, loc='upper left')
	plt.savefig(fig_name + '.png')
'''
def my_func1(x):	
	y = K.relu(x) - K.square(K.relu(-x)) / (1 + K.square(K.relu(-x)))
	return y

def my_func2(x):	
	y = K.relu(x) - K.sqrt(K.relu(-x)) / (1 + K.sqrt(K.relu(-x)))
	return y
'''

def load(name):
	import pickle
	f = open(name, 'wb')
	data = pickle.load(f)
	f.close()
	return data

def get_history(my_func, epochs=30, lr=0.001, early_stop=False):
	batch_size = 128
	num_classes = 10
	epochs = epochs

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	img_rows, img_cols = 28, 28

	if K.image_data_format() == 'channels_first':
			x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
			x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
			input_shape = (1, img_rows, img_cols)
	else:
			x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
			x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
			input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
				#activation=my_func,
				padding='valid',
				input_shape=input_shape))
	model.add(Lambda(my_func))
	model.add(Conv2D(32, kernel_size=(3, 3),
				#activation=my_func
				))
	model.add(Lambda(my_func))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128,
					#activation=my_func
					))
	model.add(Lambda(my_func))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	adam = keras.optimizers.Adam(lr=lr)

	model.compile(loss=keras.losses.categorical_crossentropy,
								optimizer=adam,
								metrics=['accuracy'])

	save_path = 'model/model0301(1).h5'

	callbacks = [
	    ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=0),
	]
	if early_stop:
		callbacks.append(EarlyStopping(monitor='val_loss', patience=5, verbose=0))

	history = model.fit(x_train, y_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=1,
						callbacks=callbacks,
						validation_data=(x_test, y_test))

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return history
#print(history.history.keys())

'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

'''
plt.figure()
plt.plot(history_softsign.history['loss'])
plt.plot(history_sigmoid.history['loss'])
plt.plot(history_my.history['loss'])
plt.plot(history_my1.history['loss'])
plt.plot(history_my2.history['loss'])
plt.title('train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['softsign', 'sigmoid', 'my_fun1', 'my_fun2', 'my_fun3'], loc='upper left')

plt.savefig('train_loss_comp_all.png')
'''