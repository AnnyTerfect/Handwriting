#-*- coding: utf8 -*-
import numpy as np

def model_evaluate(model, x_test, y_test, num_per_clu=10):
	global _fx_test, _y_test

	ly = len(y_test)
	num_cluster = int(len(y_test) / num_per_clu)
	num_cor = 0

	fx_test = list(model.predict(x_test))
	y_test = list(np.argmax(y_test, axis=1))

	_fx_test = []
	_y_test = []
	while len(y_test):
		del_list = []
		__fx_test = []

		for i, y in enumerate(y_test):
			if y == y_test[0]:
				del_list.append(i)
				__fx_test.append(fx_test[i])
				if len(__fx_test) == num_per_clu:
					break
		_y_test.append(y)
		_fx_test.append(__fx_test)

		del_list.sort(reverse=True)
		for i in del_list:
			del y_test[i]
			del fx_test[i]	

	for i, cluster in enumerate(_fx_test):
		cat = np.argmax(np.sum(np.array(cluster), axis=0))
		if cat == _y_test[i]:
			num_cor += 1

	accurate = 100.0 * num_cor / num_cluster
	#print('correct: {} of {}'.format(num_cor, num_cluster))
	#print('accurate: {}%'.format(accurate))

	return [accurate, num_cor, num_cluster]