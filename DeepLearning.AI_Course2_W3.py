"""
Skills Covered:
1) Transfer Learning (Using features from a pretrained model)
2) Adjusting input shape
3) Extracting zip and creating directories
4) Data augmentation with ImageGenerator object
5) CNN
"""
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

local_weights_file = 'C://Users//User//repos//tfdev//data//inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Initialize the base model
# Set the input shape and remove dense layers
# For images, we can use 150x150x3, but this may vary
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(local_weights_file)

# Freeze the weights of the layers.
for layer in pre_trained_model.layers:
    layer.trainable = False

# # Print summary
# pre_trained_model.summary()

# Choose `mixed_7` as the last layer of your base model
# This is the last convolutional layer, we will add our own dense layers
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for binary classification
x = layers.Dense(1, activation='sigmoid')(x)

# Append the dense network to the base model
model = Model(pre_trained_model.input, x)

# Print the model summary. See your dense network connected at the end.
model.summary()

# Set the training parameters
# Use binary crossentropy for binary class data
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define our example directories and files
base_dir = 'C://Users//User//repos//tfdev//data//tmp'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

# Extract the archive
os.mkdir('C://Users//User//repos//tfdev//data//tmp')
zip_ref = zipfile.ZipFile('C://Users//User//repos//tfdev//data//dogs-vs-cats.zip', 'r')
zip_ref.extractall('C://Users//User//repos//tfdev//data//tmp')
zip_ref.close()

os.mkdir('C://Users//User//repos//tfdev//data//tmp//tmp_train')
zip_ref_train = zipfile.ZipFile('C://Users//User//repos//tfdev//data//tmp//train.zip', 'r')
zip_ref_train.extractall('C://Users//User//repos//tfdev//data//tmp//tmp_train')
zip_ref_train.close()

# os.mkdir('C://Users//User//repos//tfdev//data//tmp//tmp_test')
# zip_ref_test = zipfile.ZipFile('C://Users//User//repos//tfdev//data//tmp//test1.zip', 'r')
# zip_ref_test.extractall('C://Users//User//repos//tfdev//data//tmp//tmp_test')
# zip_ref_test.close()

# Set up directories and subdirectories
base_dir_update = base_dir + '//catsvdogs'
os.mkdir(base_dir_update)
train_dir = base_dir_update + '//train'
test_dir = base_dir_update + '//test'
train_cats_dir = train_dir + '//cats'
train_dogs_dir = train_dir + '//dogs'
test_cats_dir = test_dir + '//cats'
test_dogs_dir = test_dir + '//dogs'
os.mkdir(train_dir)
os.mkdir(test_dir)
os.mkdir(train_dogs_dir)
os.mkdir(train_cats_dir)
os.mkdir(test_dogs_dir)
os.mkdir(test_cats_dir)

# Define train/test split
train_pct = 0.8
tmp_train_dir = 'C://Users//User//repos//tfdev//data//tmp//tmp_train'
files = os.listdir(tmp_train_dir + '//train')
train_split = int(len(files) * 0.8)
train_files = files[:train_split]
test_files = files[train_split:]

# Move the training files
for file in train_files:
    if file.split('.')[0] == 'cat':
        shutil.move(tmp_train_dir + '//train//' + file, train_cats_dir + '//' + file)
    elif file.split('.')[0] == 'dog':
        shutil.move(tmp_train_dir + '//train//' + file, train_cats_dir + '//' + file)

# Move the test files
for file in test_files:
    if file.split('.')[0] == 'cat':
        shutil.move(tmp_train_dir + '//train//' + file, test_cats_dir + '//' + file)
    elif file.split('.')[0] == 'dog':
        shutil.move(tmp_train_dir + '//train//' + file, test_cats_dir + '//' + file)


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Note that the test data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# Flow validation images in batches of 20 using test_datagen generator
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  target_size=(150, 150))

# Train the model.
history = model.fit(train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=100,
                    epochs=20,
                    validation_steps=50,
                    verbose=2)

# Evaluate results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

# Remove the base_dir
shutil.rmtree(base_dir)