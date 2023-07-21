import numpy as np
import tensorflow as tf

# Week 1  - Hello World
# model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
# xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# model.fit(xs, ys, epochs=500)
# print(model.predict([10.0]))

# Week 2 - Computer Vision
# Callback class will stop training when a desired level of accuracy is reached.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.9): # Experiment with changing this value
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)
train_images = train_images / 255.0  # NN performs better on normalized data
test_images = test_images / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),  # take the (28, 28) grayscale matrix and flatten into an (784,) array
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Middle layer has 128 "nodes"
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # The number of neurons in the final layer = number of classes
    ])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# If number of epochs is too high, loss can increase ("overfitting")
with tf.device('/gpu:0'):  # Control whether to use CPU or GPU
    model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
    model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])  # Outputs a list of probabilities corresponding to each class.
print(test_labels[0])