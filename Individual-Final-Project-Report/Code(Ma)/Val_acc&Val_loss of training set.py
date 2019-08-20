import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])


history = model.fit(x_train, y_train, epochs=10,validation_split=0.1, batch_size=100, verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

