# Convolutional Neural Network

# import keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
# from IPython.display import display
# from PIL import Image
import numpy as np
import os

# initialize CNN
classifier = Sequential()

# add convolutional layers
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))

# perform max pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# flatter the image to a linear array
classifier.add(Flatten())

# connect CNN to a NN
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# then compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=20,
    validation_data=test_set,
    validation_steps=800
)

# save the model
# exists = os.path.isfile('model.hdf5')
# if exists:
#     classifier.load_weights('model.hdf5')
# else:
#     classifier.save('model.hdf5')
classifier.save('model.hdf5')

# test a random dog picture
test_image = image.load_img('jake.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
# training_set.class_indices
if result [0][0] >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print('jake.jpg is a ', prediction)
