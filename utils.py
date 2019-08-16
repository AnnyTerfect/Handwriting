#-*- coding: utf8 -*-
import cv2
import numpy as np
import os
import pickle
import multiprocessing as mp

def img_enpower(img, direction=0):
	if direction == 0:
		di1, di2, dj1, dj2 = 0, 0, 20, 0

	if len(img.shape) == 3:
		img = img.reshape((img.shape[0], img.shape[1]))
	h, w = img.shape

	timg = img[di1: h - di2, dj1: w - dj2]
	img = np.zeros((h, w))

	ci = int((di1 + di2) / 2)
	cj = int((dj1 + dj2) / 2)
	img[ci: h - ci, cj: w - cj] = timg

	return img

def resize_cont(img, h, w):
	ori_h, ori_w, t = img.shape
	rate = min(1.0 * h / ori_h, 1.0 * w / ori_w)
	new_h = int(ori_h * rate)
	new_w = int(ori_w * rate)
	timg = cv2.resize(img, (new_w, new_h))
	
	img = np.zeros((h, w, 3))
	i = int((h - new_h) / 2)
	j = int((w - new_w) / 2)

	img[i: i + new_h, j: j + new_w] = 255 - timg

	return img

def save_sample(datas, n):
	if not(os.path.exists('./sapmle')):
		os.system('mkdir sapmle')

	for i in range(n):
		cv2.imwrite('sample/sample{}.png'.format(i + 1), datas[i] * 255)

def dump(data, file):
	f = open(file, 'wb')
	pickle.dump(data, f)
	f.close()