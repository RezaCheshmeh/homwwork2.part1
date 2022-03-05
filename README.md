# homwwork2.part1
تمرین دوم بخش یک
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
plt.imshow(x_train[7])
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train[0:10]
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
y_train[0:10]
model = Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=x_train[0].shape))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))


model.summary()
opt_rms = keras.optimizers.RMSprop(learning_rate=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt_rms,
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          epochs=25, batch_size=64, validation_data = (x_test, y_test))

