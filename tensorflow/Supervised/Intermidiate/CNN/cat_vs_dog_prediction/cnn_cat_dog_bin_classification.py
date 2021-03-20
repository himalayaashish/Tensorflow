# Unzip the file cats-and-dogs.zip

import os
import zipfile
import random
from shutil import copyfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow  as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

local_zip = '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/')
zip_ref.close()


# Definng and mapping the directories
print(len(os.listdir('/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/PetImages/Cat/')))
print(len(os.listdir('/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/PetImages/Dog/')))

to_create = [
    '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs',
    '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training',
    '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing',
    '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training/cats',
    '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training/dogs',
    '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing/cats',
    '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing/dogs'
]

for directory in to_create:
    try:
        os.mkdir(directory)
        print(directory, 'created')
    except:
        print(directory, 'failed')

# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_files = []

    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))

    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)

    shuffled = random.sample(all_files, n_files)

    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]

    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)

    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)


CAT_SOURCE_DIR = "/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/PetImages/Cat/"
TRAINING_CATS_DIR = "/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/PetImages/Dog/"
TRAINING_DOGS_DIR = "/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training/cats/')))
print(len(os.listdir('/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing/dogs/')))

# Lets plot some

train_cat_fnames = os.listdir( TRAINING_CATS_DIR )
train_dog_fnames = os.listdir( TRAINING_DOGS_DIR )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

nrows = 4
ncols = 4
pic_index = 0
fig_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(TRAINING_CATS_DIR , fname)
                for fname in train_cat_fnames[ pic_index-8:pic_index]
               ]

next_dog_pix = [os.path.join(TRAINING_DOGS_DIR, fname)
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# Creating the model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])


TRAINING_DIR = '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/training'
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
)

VALIDATION_DIR = '/home/himalaya/PycharmProjects/tensorflow/tensorflow/Intermidiate/data/cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'

)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
)



history = model.fit_generator(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator)

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.show()
#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')
plt.show()