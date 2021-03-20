import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_data,training_labels),(testing_data,testing_labels) = mnist.load_data()
print(type(training_data))
print(training_data.shape)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.9):
            print("\n Reached 90 % accyracy so cancelling training")
            self.model.stop_training=True


callbacks = myCallback()

"""
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_data[100])
print(training_data[100])
print(training_labels[100])

"""
training_data = training_data.reshape(60000,28,28,1)
training_data = training_data/255.0
testing_data = testing_data.reshape(10000,28,28,1)
testing_data = testing_data/255.0

#Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_data,training_labels,epochs=100,callbacks=[callbacks])

model.evaluate(testing_data,testing_labels)

classification = model.predict(testing_data)

print(testing_labels[0])
print("#"*30)

# Visulizing the concolution and Pooling layers

import matplotlib.pyplot as plt
f,axarr = plt.subplots(3,4)
FIRST_IMAGE = 4
SECOND_IMAGE = 7
THIRD_IMAGE = 26
CONVOLUTION_NUMBER = 4

from tensorflow.keras import models
layer_output = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input,outputs=layer_output)


for x in range(0,4):
    f1 = activation_model.predict(testing_data[FIRST_IMAGE].reshape(1,28,28,1))[x]
    axarr[0,x].imshow(f1[0,:,:,CONVOLUTION_NUMBER],cmap='inferno')
    axarr[0,x].grid(False)

    f2 = activation_model.predict(testing_data[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)

    f3 = activation_model.predict(testing_data[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)

plt.show()