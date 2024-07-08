import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Paths
train_Path = '/home/shephali/Desktop/Resnet50/train'
test_Path = '/home/shephali/Desktop/Resnet50/test'

# Set Resize variable
IMAGE_SIZE = [224, 224]

# Initialize ResNet50
resnet = ResNet50(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False
)

# Freeze all layers
for layer in resnet.layers:
    layer.trainable = False

# Useful for getting the number of output classes.
folders = glob(train_Path + '/*')

# Set the flatten layer.
x = Flatten()(resnet.output)
x = Dropout(0.5)(x)  # Add dropout to reduce overfitting

# Add the final Dense layer with softmax activation
prediction = Dense(len(folders), activation='softmax')(x)

# Create a model object
model = Model(inputs=resnet.input, outputs=prediction)

model.summary()

# Compile the model with a lower learning rate
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Use the Image Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,  # Reduced shear range
    zoom_range=0.1,   # Reduced zoom range
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_Path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_Path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Convert the generators to tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: training_set,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 224, 224, 3], [None, len(folders)])
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_set,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 224, 224, 3], [None, len(folders)])
)

# Repeat the datasets
train_dataset = train_dataset.repeat()
test_dataset = test_dataset.repeat()

# Create EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity
)

# Fit the model with early stopping
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=[early_stopping] # Add the early stopping callback
)

# Plot the Loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot the Accuracy
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# Save it as a h5 file
model.save('car_brand_clf_resnet50.h5')

# Predict on the test set
prediction = model.predict(test_set)
prediction = np.argmax(prediction, axis=1)
print(prediction)
