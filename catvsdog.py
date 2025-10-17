import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Define dataset path
data_dir = '/kaggle/input/dog-vs-cat/animals'
print(os.listdir(data_dir))

# Define subdirectories
cat_dir = os.path.join(data_dir, 'cat')
dog_dir = os.path.join(data_dir, 'dog')

# Count images per class
num_cats = len(os.listdir(cat_dir))
num_dogs = len(os.listdir(dog_dir))
print(f'Total Cats: {num_cats}, Total Dogs: {num_dogs}')

# Visualize sample images
import random
from PIL import Image

sample_cats = random.sample(os.listdir(cat_dir), 3)
sample_dogs = random.sample(os.listdir(dog_dir), 3)

fig, axes = plt.subplots(2, 3, figsize=(10,6))
for i, img_name in enumerate(sample_cats):
    img_path = os.path.join(cat_dir, img_name)
    axes[0, i].imshow(Image.open(img_path))
    axes[0, i].set_title('Cat')
    axes[0, i].axis('off')
for i, img_name in enumerate(sample_dogs):
    img_path = os.path.join(dog_dir, img_name)
    axes[1, i].imshow(Image.open(img_path))
    axes[1, i].set_title('Dog')
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()

# Define parameters
img_size = (150, 150)
batch_size = 32

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

# Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=12,
    callbacks=[early_stop, reduce_lr]
)

# Plot accuracy and loss
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.title('Accuracy');

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss');
plt.show()

# Evaluate on validation set
val_gen.reset()
preds = (model.predict(val_gen) > 0.5).astype(int)
print(classification_report(val_gen.classes, preds, target_names=['Cat','Dog']))