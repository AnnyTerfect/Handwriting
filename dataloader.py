#-*- coding: utf8 -*-
import os
import numpy as np
import cv2
from utils import *
from time import time
from tqdm import tqdm
import sys
import keras
from keras import backend as K

def myprint(content, end=''):
    print(content, end=end)

def load(data='train test', train_size=2000, test_size=300, _writer_num=100, size=(81, 78)):
	global now, data_num, data_total_num, writer_num, H, W
	global train, test

	now = time()
	data_num = 0
	data_total_num = 0
	writer_num = _writer_num
	train = list()
	test = list()

	H, W = size

	load_opr = ''
	if 'train' in data:
		data_total_num += train_size * writer_num
		load_opr += 'loadbatch(\'train\', 1, 2000)' + '\n'
		load_opr += 'myprint(\'\\n训练集加载成功用时%ss\'%(time() - now), end=\'\\n\')' + '\n'
	if 'test' in data:
		data_total_num += test_size * writer_num
		load_opr += 'loadbatch(\'test\', 1, 300)' + '\n'
		load_opr += 'myprint(\'\\n测试集加载成功用时%ss\'%(time() - now), end=\'\\n\')' + '\n'

	exec(load_opr)
	
	x_train = [x[0] for x in train]
	y_train = [x[1] for x in train]
	x_test = [x[0] for x in test]
	y_test = [x[1] for x in test]


	x_train = np.array(x_train)
	x_test = np.array(x_test)
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, H, W)
		x_test = x_test.reshape(x_test.shape[0], 1, H, W)
	else:
		x_train = x_train.reshape(x_train.shape[0], H, W, 1)
		x_test = x_test.reshape(x_test.shape[0], H, W, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, writer_num)
	y_test = keras.utils.to_categorical(y_test, writer_num)

	return (x_train, y_train), (x_test, y_test)

def loadbatch(datatype, l, u):
	global data_num
	dir = ''

	if datatype == 'train':
		num_base = 0
		dir_base = '/TRAIN_DATA/Dataset_png'
		data_index = 0
		operate = 'train.append((img, i - 1))' + '\n'
		#operate += 'train.append((img_enpower(img, 0), i - 1))'
	elif datatype == 'test':
		num_base = 3000
		dir_base = '/TEST_DATA/Dataset_png'
		data_index = 1
		operate = 'test.append((img, i - 1))' + '\n'
		#operate += 'test.append((img_enpower(img, 0), i - 1))'
	else:
		return

	for j in range(l + num_base,  u + 1 + num_base):
		r = list(range(1, writer_num + 1))
		np.random.shuffle(r)
		for i in r:
			data_num += 1
			
			cost_time = time() - now
			remain_time = cost_time / (data_num / data_total_num) - cost_time
			remain_time = int(remain_time * 10) / 10.0
			msg = 'loading {0} of {1} cost {2}s remain {3}s \t'\
			.format(int(data_num), int(data_total_num), int(cost_time), remain_time)
			myprint(msg, end = '\r')

			dir = os.getcwd() + dir_base + '%d/'%(i)

			img = cv2.imread(dir + str(j) + '.png')
			try:
				while(not(img.size)):
					img = cv2.imread(dir + str(j + 1) + '.png')
			except AttributeError:
				print('error')
				print('i = %d'%i)
				print('j = %d'%j)
				continue
			img = resize_cont(img, H, W)

			img = np.average(img, axis=-1) / 255

			exec(operate)
			