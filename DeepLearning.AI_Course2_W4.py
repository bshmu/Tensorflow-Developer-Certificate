"""
Skills Covered:
1) Adjusting input shape (expanding the dimension)
2) Loading CSV using pandas
3) ImageDataGenerator
4) MultiClass Classification -- use 'sparse_categorical_crossentropy' for the loss
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# Callback class will stop training when a desired level of accuracy is reached.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.97): # Experiment with changing this value
            print("\nReached 97% accuracy so cancelling training!")
            self.model.stop_training = True

path = 'C:\\Users\\User\\repos\\tfdev\\data'

def get_data(filename):
    with open(filename) as training_file:
        file_df = pd.read_csv(filename)
        labels = file_df['label'].to_numpy()
        images = file_df[file_df.columns[1:]].to_numpy()
        images = images.reshape(images.shape[0], 28, 28)
        return images, labels

path_sign_mnist_train = path + '\\sign_mnist_train.csv'
path_sign_mnist_test = path + '\\sign_mnist_test.csv'
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)
#
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

# Instantiate ImageDataGenerator object with data augmentation parameters
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

print("Training images:", training_images.shape)
print("Testing images:", testing_images.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')])

# model.summary()

model.compile(optimizer=tf.optimizers.RMSprop(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with tf.device('/gpu:0'):  # Control whether to use CPU or GPU
    # Fit Model
    history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=1),
                                  steps_per_epoch=len(training_images)//10,
                                  epochs=3,
                                  validation_data=test_datagen.flow(testing_images, testing_labels, batch_size=1),
                                  validation_steps=len(testing_images)//10,
                                  callbacks=[myCallback()])
    model.evaluate(testing_images, testing_labels, verbose=0)

# Plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()