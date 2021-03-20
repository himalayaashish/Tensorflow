import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.9):
            print("\n Reached 90% accuracy so canceling the training")
            self.model.stop_training = True


mnist = tf.keras.datasets.fashion_mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train,X_test = X_train/255.0, X_test/255.0

callbacks = myCallback()

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                             tf.keras.layers.Dense(512,activation = tf.nn.relu),
                             tf.keras.layers.Dense(10,activation=tf.nn.softmax)
                             ])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(X_train,y_train,epochs=20,callbacks=[callbacks])

print(model.evaluate(X_test,y_test))