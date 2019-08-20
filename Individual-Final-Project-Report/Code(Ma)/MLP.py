import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential


mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Reshape to 28 x 28 pixels = 784 features
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Convert into greyscale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Convert target classes （0-9）to categorical ones
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# second method: normalize the train&test numpy array
##x_train = keras.utils.normalize(x_train,axis=1)
#x_test = keras.utils.normalize(x_test,axis=1)


model = Sequential()

# Second way to flattne the input data: model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))

model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy', # change to sparse_categorical_crossentropy if choose the normalize way
             metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 100, verbose = 1, epochs = 10)
model.summary()

val_loss,val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)