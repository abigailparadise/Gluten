
from tensorflow.keras.applications import ResNet50V2

import os
import random
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import legacy as optimizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import warnings


def split_and_sort_images(source_path, category, destination_path, train_ratio=0.80, val_ratio=0.1, test_ratio=0.1,
                          seed=42):
    if not os.path.exists(source_path):
        print(f"Source path '{source_path}' does not exist.")
        return

    train_folder = os.path.join(destination_path, 'train', f'train_{category}')
    val_folder = os.path.join(destination_path, 'val', f'val_{category}')
    test_folder = os.path.join(destination_path, 'test', f'test_{category}')

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    imgs_list = [filename for filename in os.listdir(source_path) if os.path.splitext(filename)[-1] in image_extensions]
    random.seed(seed)
    random.shuffle(imgs_list)

    total_images = len(imgs_list)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size

    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    for i, filename in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        elif i < train_size + val_size:
            dest_folder = val_folder
        else:
            dest_folder = test_folder
        shutil.copy(os.path.join(source_path, filename), os.path.join(dest_folder, filename))
    print(f"Successfully sorted images for category '{category}' into train, val, and test folders.")




sorted_images_path = r"C:\Users\Abigail Paradise Vit\Downloads\DATA SETS\COMBINE2"
gluten_path = r"C:\Users\Abigail Paradise Vit\Desktop\train_gluten_combine - Copy"
gluten_free_path =r"C:\Users\Abigail Paradise Vit\Desktop\train_gluten_free_combine - Copy"

# Split and sort the datasets
#split_and_sort_images(gluten_path, 'gluten', sorted_images_path)
#split_and_sort_images(gluten_free_path, 'gluten_free', sorted_images_path)


# Image data generator with augmentation for training and rescale for validation and test
train_image_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.2
)

val_test_gen = ImageDataGenerator(rescale=1/255)

# Training, validation, and test generators
train_image_gen = train_image_gen.flow_from_directory(
    r'C:\Users\Abigail Paradise Vit\Downloads\DATA SETS\COMBINE2\train',
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary'
)

val_image_gen = val_test_gen.flow_from_directory(
   r'C:\Users\Abigail Paradise Vit\Downloads\DATA SETS\COMBINE2\val',
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary'
)

test_image_gen = val_test_gen.flow_from_directory(
r'C:\Users\Abigail Paradise Vit\Downloads\DATA SETS\COMBINE2\test',
    target_size=(150, 150),
    batch_size=34,
    class_mode='binary',
    shuffle=False  # Ensure the order of images is the same
)

base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


# Model definition using VGG16


# Add new top layers for binary classification
x = base_model.output
x = Flatten()(x)  # Flatten the output tensor of the last convolutional block
x = Dense(128, activation='relu')(x)  # Fully connected layer with 128 units and ReLU activation
x = Dropout(0.5)(x)  # Add Dropout for regularization
x = Dense(1, activation='sigmoid')(x)  # Final layer with 1 unit and Sigmoid activation for binary classification

# Create the new model
model = Model(inputs=base_model.input, outputs=x)

# Optionally, freeze the layers of the base VGG16 model to retain pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.00001),
              metrics=['accuracy'])

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Calculate steps per epoch
train_steps = len(train_image_gen)
val_steps = len(val_image_gen)

# Training the model with validation data
warnings.filterwarnings('ignore')

# Train the model
results = model.fit(
    train_image_gen,
    epochs=100,
    steps_per_epoch=train_steps,
    validation_data=val_image_gen,
    validation_steps=val_steps,
    callbacks=[early_stop, reduce_lr]
)

# Final evaluation on the test set
test_steps = len(test_image_gen)
evaluation = model.evaluate(test_image_gen, steps=test_steps)
print(f'Test Loss: {evaluation[0]}')
print(f'Test Accuracy: {evaluation[1]}')

# Reinitialize the test generator to ensure it starts from the beginning
test_image_gen = val_test_gen.flow_from_directory(
    r'C:\Users\Abigail Paradise Vit\Downloads\DATA SETS\COMBINE2\test',
    target_size=(150, 150),
    batch_size=34,
    class_mode='binary',
    shuffle=False  # Ensure the order of images is the same
)

# Predictions and metrics
predictions = model.predict(test_image_gen, steps=test_steps)
predictions = np.where(predictions > 0.5, 1, 0)

#save model
model.save('saved_model.h5')

# Accuracy, Precision, Recall, F1-Score
accuracy = accuracy_score(test_image_gen.classes, predictions)
precision = precision_score(test_image_gen.classes, predictions)
recall = recall_score(test_image_gen.classes, predictions)
f1 = f1_score(test_image_gen.classes, predictions)

# Classification report
class_report = classification_report(test_image_gen.classes, predictions, target_names=['Gluten Free', 'Gluten Containing'])
print(class_report)

# Display results
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')


# Confusion matrix
conf_matrix = confusion_matrix(test_image_gen.classes, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Gluten Free', 'Gluten Containing'], yticklabels=['Gluten Free', 'Gluten Containing'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()