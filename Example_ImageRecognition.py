import os
import shutil
import tensorflow as tf
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

dir = r'C:\Users\User\repos\tfdev\data\tmp'
if os.path.isdir(dir):
    shutil.rmtree(dir)
else:
    pass
os.mkdir(dir)
train_dir = dir + r'\train'
test_dir = dir + r'\test'
os.mkdir(train_dir)
os.mkdir(test_dir)

local_zip = r'C:\Users\User\repos\tfdev\data\horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(train_dir)
zip_ref.close()

local_zip = r'C:\Users\User\repos\tfdev\data\validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(test_dir)
zip_ref.close()

# Set up directories
train_horse_dir = train_dir + '\horses'
train_human_dir = train_dir + '\humans'
test_horse_dir = test_dir + '\horses'
test_human_dir = test_dir + '\humans'

# Define train/validation split
train_files = []
for dir in [train_horse_dir, train_human_dir]:
    train_files += os.listdir(dir)
print(len(train_files), 'training files')

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=5,
                                                    class_mode='binary',
                                                    target_size=(300, 300))
validation_generator = validation_datagen.flow_from_directory(test_dir,
                                                              batch_size=5,
                                                              class_mode='binary',
                                                              target_size=(300, 300))

# Define model
pre_trained_model = VGG16(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
for layer in pre_trained_model.layers:
    layer.trainable = False

# model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
#                                     tf.keras.layers.MaxPooling2D(2, 2),
#                                     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#                                     tf.keras.layers.MaxPooling2D(2, 2),
#                                     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#                                     tf.keras.layers.MaxPooling2D(2, 2),
#                                     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#                                     tf.keras.layers.MaxPooling2D(2, 2),
#                                     tf.keras.layers.Dropout(0.2),
#                                     tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(512, activation='relu'),
#                                     tf.keras.layers.Dense(128, activation='relu'),
#                                     tf.keras.layers.Dense(1, activation='sigmoid')
#                                     ])

x = tf.keras.layers.Flatten()(pre_trained_model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(pre_trained_model.input, x)

model.compile(optimizer=tf.optimizers.RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=20,
                    validation_steps=50,
                    verbose=1)

# Remove the base_dir
shutil.rmtree(dir)

print('accuracy:', "{:.2%}".format(history.history['accuracy'][-1]))
print('val_accuracy:', "{:.2%}".format(history.history['val_accuracy'][-1]))

