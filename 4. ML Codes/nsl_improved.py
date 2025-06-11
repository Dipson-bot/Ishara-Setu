import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn.utils.class_weight import compute_class_weight

# Define your labels
index_to_folder = {
    0: "क", 1: "ख", 2: "ग", 3: "घ", 4: "ङ",
    5: "च", 6: "छ", 7: "ज", 8: "झ", 9: "ञ",
    10: "ट", 11: "ठ", 12: "ड", 13: "ढ", 14: "ण",
    15: "त", 16: "थ", 17: "द", 18: "ध", 19: "न",
    20: "प", 21: "फ", 22: "ब", 23: "भ", 24: "म",
    25: "य", 26: "र", 27: "ल", 28: "व", 29: "श",
    30: "ष", 31: "स", 32: "ह", 33: "क्ष", 34: "त्र", 35: "ज्ञ"
}

train_dir = 'data/train'
test_dir = 'data/test'

# Data augmentation without horizontal flip
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir,
                              target_size=(100,100),
                              batch_size=32,
                              color_mode='grayscale',
                              class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir,
                            target_size=(100,100),
                            batch_size=32,
                            color_mode='grayscale',
                            class_mode='categorical')

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(100,100,1),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36,activation='softmax'))

# Use Adam optimizer with a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0005)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_improved.h5', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

train_labels = train_data.classes
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

history = model.fit(train_data, 
                    epochs=100,
                    steps_per_epoch=3000//train_data.batch_size,
                    validation_data=test_data, 
                    validation_steps=1000//test_data.batch_size,
                    class_weight=class_weights,
                    callbacks=[early_stop, checkpoint, reduce_lr])

model.save('finalNsl_improved.h5')

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

img_folder = 'data/test'
all_classes = os.listdir(img_folder)
random_class = random.choice(all_classes)
img_files = [f for f in os.listdir(os.path.join(img_folder, random_class)) if f.endswith('.jpg') or f.endswith('.png')]
rand_img = random.choice(img_files)
img_path = os.path.join(img_folder, random_class, rand_img)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (100, 100))
plt.imshow(img, cmap='gray')
plt.show()

img = img.reshape(1, 100, 100, 1).astype('float32') / 255

pred = model.predict(img)
predicted_class_index = np.argmax(pred)
predicted_folder = index_to_folder[int(predicted_class_index)]
print(f"The predicted folder is: {predicted_folder}")
