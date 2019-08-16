from dataloader import load
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import dump
import sys
#from keras.datasets import mnist
from model import dcnn
from time import time

batch_size = 128
num_classes = 299
epochs = 1000
size = (81, 78)

try:
	x_train.shape
	y_train.shape
	x_test.shape
	y_test.shape
except NameError:
	now = time()
	(x_train, y_train), (x_test, y_test) = load(size=size, _writer_num=num_classes)
	print('data loaded')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = dcnn(input_shape=(size[0], size[1], 1), num_classes=num_classes)

save_path = 'model/model04052059.h5'
log_path = './logs/train_logs'

earlystopping = EarlyStopping(
	monitor='val_loss', 
	patience=5, 
	verbose=0
)
modelcheckpoint = ModelCheckpoint(
	save_path, 
	monitor='val_loss', 
	save_best_only=True, 
	verbose=0
)

callbacks = [
    modelcheckpoint,
    earlystopping
]

history = model.fit(
	x_train,
	y_train,
	batch_size=batch_size,
	epochs=500,
	verbose=1,
	callbacks=callbacks,
	validation_data=(x_test, y_test)
)

dump(history, 'history')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
