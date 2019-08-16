#-*- coding: utf8 -*-
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Lambda
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras import backend as K

def Conv_Layer(filters, kernel_size, input_shape=None):
	if input_shape:
		return Conv2D(
			filters=filters,
			kernel_size=kernel_size,
			activation='relu',
			strides=1,
			padding='same',
			input_shape=input_shape
		)
	else:
		return Conv2D(
			filters=filters,
			kernel_size=kernel_size,
			activation='relu',
			strides=1,
			padding='same',
		)

def Pooling_Layer(pool_size, strides):
	return MaxPooling2D(
		pool_size=pool_size,
		strides=strides
	)

def dcnn(input_shape=(81, 78, 1), num_classes=100):
	model = Sequential()

	model.add(Conv_Layer(filters=24, kernel_size=(5, 5), input_shape=input_shape))
	model.add(Pooling_Layer(pool_size=(3, 3), strides=2))
	model.add(Conv_Layer(filters=32, kernel_size=(5, 5)))
	model.add(Pooling_Layer(pool_size=(3, 3), strides=2))
	model.add(Conv_Layer(filters=64, kernel_size=(5, 5)))
	model.add(Pooling_Layer(pool_size=(3, 3), strides=2))

	model.add(Flatten())

	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(
		loss=categorical_crossentropy,
		optimizer=sgd,
		metrics=['accuracy']
	)

	return model

def Cp_Layer(x):
	y = Conv_Layer(filters=24, kernel_size=(5, 5))(x)
	y = Pooling_Layer(pool_size=(3, 3), strides=2)(y)
	y = Conv_Layer(filters=48, kernel_size=(5, 5))(y)
	y = Pooling_Layer(pool_size=(3, 3), strides=2)(y)
	y = Conv_Layer(filters=96, kernel_size=(5, 5))(y)
	y = Pooling_Layer(pool_size=(3, 3), strides=2)(y)
	y = Flatten()(y)
	return y

def multidcnn(input_shape=(81, 78, 1), num_classes=100):
	inputs = [Input(shape=input_shape) for i in range(10)]

	y = [Cp_Layer(x) for x in inputs]
	y = Concatenate()(y)
	y = Dense(4096, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(num_classes, activation='softmax')(y)

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	model = Model(inputs=inputs, outputs=y)

	model.compile(
		loss=categorical_crossentropy,
		optimizer=sgd,
		metrics=['accuracy']
	)

	return model

def sse((x1, x2)):
	return K.square(x1 - x2)

def similarity(input_shape=(81, 78, 1)):
	inputs = [Input(shape=input_shape), Input(shape=input_shape)]
	y = [Cp_Layer(x) for x in inputs]
	y = Lambda(sse)(y)
	y = Dense(100, activation='relu')(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(inputs=inputs, outputs=y)

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(
		loss='binary_crossentropy',
		optimizer=sgd,
		metrics=['accuracy']
	)

	return model