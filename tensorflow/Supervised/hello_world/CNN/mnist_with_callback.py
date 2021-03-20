import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_data,training_labels),(testing_data,testing_labels) = mnist.load_data()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.9):
            print("\n Reached 90 % accyracy so cancelling training")
            self.model.stop_training=True


callbacks = myCallback()
#print(type(training_data))

#print(training_data.shape)
"""
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_data[100])
print(training_data[100])
print(training_labels[100])

"""
training_data = training_data/255.0
testing_data = testing_data/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_data,training_labels,epochs=100,callbacks=[callbacks])

model.evaluate(testing_data,testing_labels)

classification = model.predict(testing_data)

print(testing_labels[0])
