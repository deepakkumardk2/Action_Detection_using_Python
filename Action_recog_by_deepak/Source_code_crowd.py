import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, SimpleRNN
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --------------------------- Configuration ---------------------------
train_path = 'maan.mp4'  # Replace with the actual path
test_path = 'maan.mp4'    # Replace with the actual path

sequence_length = 60  # Number of frames in each video sequence
frame_size = (240, 240, 3)  # Resize frames to 240x240 with 3 channels (RGB)
batch_size = 8
num_classes = 10  # Replace with the number of classes in your dataset

# --------------------------- Data Generator ---------------------------
class VideoFrameGenerator(tf.keras.utils.Sequence):
    def __init__(self, video_dir, batch_size, sequence_length, frame_size, num_classes):
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.num_classes = num_classes

        # Get list of video directories
        self.video_paths = []
        self.labels = []
        for label, folder in enumerate(sorted(os.listdir(video_dir))):
            folder_path = os.path.join(video_dir, folder)
            if os.path.isdir(folder_path):
                for video_folder in os.listdir(folder_path):
                    video_path = os.path.join(folder_path, video_folder)
                    if os.path.isdir(video_path):
                        self.video_paths.append(video_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.video_paths) // self.batch_size

    def __getitem__(self, index):
        batch_paths = self.video_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = [], []
        for video_path, label in zip(batch_paths, batch_labels):
            frames = sorted(os.listdir(video_path))[:self.sequence_length]
            video_frames = []

            for frame in frames:
                frame_path = os.path.join(video_path, frame)
                if os.path.isfile(frame_path):
                    img = load_img(frame_path, target_size=self.frame_size[:2])
                    img_array = img_to_array(img) / 255.0
                    video_frames.append(img_array)

            # Ensure the correct number of frames
            if len(video_frames) == self.sequence_length:
                X.append(video_frames)
                y.append(label)

        return np.array(X), tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

# Instantiate data generators
train_generator = VideoFrameGenerator(train_path, batch_size, sequence_length, frame_size, num_classes)
test_generator = VideoFrameGenerator(test_path, batch_size, sequence_length, frame_size, num_classes)

# --------------------------- Model Architecture ---------------------------
inputs = Input(shape=(sequence_length, *frame_size))
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
x = TimeDistributed(MaxPooling2D((2, 2)))(x)
x = TimeDistributed(Flatten())(x)
x = Reshape((sequence_length, -1))(x)

x = SimpleRNN(128, activation='relu', return_sequences=False)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --------------------------- Callbacks ---------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# --------------------------- Training ---------------------------
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# --------------------------- Evaluation ---------------------------
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --------------------------- Metrics Calculation ---------------------------
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.concatenate([np.argmax(test_generator[i][1], axis=1) for i in range(len(test_generator))])

# F1 Score
f1 = f1_score(true_labels, predicted_labels, average='macro')
print(f"F1 Score: {f1:.4f}")

# AUC Score
true_labels_onehot = tf.keras.utils.to_categorical(true_labels, num_classes=num_classes)
auc = roc_auc_score(true_labels_onehot, predictions, multi_class='ovr')
print(f"AUC Score: {auc:.4f}")

# --------------------------- Plot Results ---------------------------
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()