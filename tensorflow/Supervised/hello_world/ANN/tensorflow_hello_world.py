import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Creating the simplest possible neural network with one layer [Dense (unit=1)]
# and with one neuron [input_shape=[1]]
model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape = [1])])

# Compile the model
model.compile(optimizer='sgd',loss='mean_squared_error')

# Our data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#Training the neural network
model.fit(xs,ys,epochs=500)

#Testing the moodel
print("#"*30)
print(model.predict([10.0]))
