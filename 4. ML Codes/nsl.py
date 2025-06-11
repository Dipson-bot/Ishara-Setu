
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import pickle

# Romanized Devanagari class labels
index_to_label = [
    "ka", "kha", "ga", "gha", "nga",
    "cha", "chha", "ja", "jha", "yna",
    "tta", "ttha", "dda", "ddha", "nna",
    "ta", "tha", "da", "dha", "na",
    "pa", "pha", "ba", "bha", "ma",
    "ya", "ra", "la", "wa",
    "sha", "ssha", "sa", "ha",
    "ksha", "tra", "gya","khali"
]

label_to_index = {label: i for i, label in enumerate(index_to_label)}

# Load pickled data
def load_pickle_data(prefix):
    with open(f"nsldata/{prefix}_images", "rb") as f:
        images = np.array(pickle.load(f))
    with open(f"nsldata/{prefix}_labels", "rb") as f:
        labels = np.array(pickle.load(f))
    return images, labels

train_images, train_labels = load_pickle_data("train")
test_images, test_labels = load_pickle_data("test")

# Preprocess images
train_images = np.array([cv2.resize(img, (128, 128)) for img in train_images])
test_images = np.array([cv2.resize(img, (128, 128)) for img in test_images])

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_images = train_images.reshape(train_images.shape[0], 128, 128, 1)
test_images = test_images.reshape(test_images.shape[0], 128, 128, 1)

# Convert labels to categorical
train_labels_cat = keras.utils.to_categorical(train_labels, num_classes=len(index_to_label))
test_labels_cat = keras.utils.to_categorical(test_labels, num_classes=len(index_to_label))

# Define CNN Model
model = Sequential([
    Input(shape=(128, 128, 1)),

    Conv2D(32, 3, padding='same', activation='relu'),
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(2),

    Conv2D(64, 3, padding='same', activation='relu'),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2),

    Conv2D(128, 3, padding='same', activation='relu'),
    Conv2D(128, 3, activation='relu'),
    MaxPool2D(2),

    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(256, 3, activation='relu'),
    MaxPool2D(2),

    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(512, 3, activation='relu'),
    MaxPool2D(2),

    Dropout(0.25),
    Flatten(),
    Dense(1500, activation='relu'),
    Dropout(0.4),
    Dense(len(index_to_label), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

history = model.fit(
    train_images, train_labels_cat,
    batch_size=32,
    epochs=10,
    validation_data=(test_images, test_labels_cat),
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save('finalNsl.h5')

# Plot Accuracy and Loss
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.title("Model Loss")
plt.show()

# Predict a random test image
random_idx = random.randint(0, test_images.shape[0] - 1)
img = test_images[random_idx].reshape(128, 128)
actual_label = index_to_label[test_labels[random_idx]]

plt.imshow(img, cmap='gray')
plt.title(f"Actual: {actual_label}")
plt.show()

img_input = test_images[random_idx].reshape(1, 128, 128, 1).astype('float32')
pred = model.predict(img_input)
predicted_class_index = np.argmax(pred)
predicted_label = index_to_label[predicted_class_index]

print(f"Predicted Label: {predicted_label}")
