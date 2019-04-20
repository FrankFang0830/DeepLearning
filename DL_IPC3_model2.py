# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks as ck
import tensorflow as tf
K.set_image_dim_ordering('th')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_train=X_train[1:6000]

X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# convert the y values into 0 and 1
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_train=y_train[1:6000]
num_classes = y_test.shape[1]
# Create the model
#Applying convolution, max-pooling, flatten and a dense layer sequentially.
# convolution requires 3D input(height, width, color_channels_depth).


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2)) #prevent overfitting , we set a probability to ignore some feature or neuron
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
#tensorboard to visualize

tbCallBack= ck.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32,callbacks=[tbCallBack])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#test set
single_test=X_test[0:4,:]

predicted=model.predict(single_test)
predicted1=predicted.tolist()
print (predicted1[0].index(max(predicted1[0])))
print (predicted1[1].index(max(predicted1[1])))
print (predicted1[2].index(max(predicted1[2])))
print (predicted1[3].index(max(predicted1[3])))

print(y_test[0:4])

