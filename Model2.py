import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras import layers, models
from keras.callbacks import ModelCheckpoint

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract keypoints from an image using MediaPipe
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).flatten()  # Flatten to a 1D array of 99 values
    else:
        return None

# Function to prepare the dataset
def prepare_data(dataset_dir):
    keypoints = []
    labels = []

    for subdir in os.listdir(dataset_dir):
        subdir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing class {subdir}...")
            for image_name in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, image_name)

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error: Could not load image {image_path}")
                    continue  # Skip if image couldn't be loaded

                keypoint = extract_keypoints(image)
                if keypoint is not None:
                    keypoints.append(keypoint)
                    labels.append(subdir)  # Use subdir name as the label (class name)

    # Convert keypoints and labels to numpy arrays
    keypoints = np.array(keypoints)
    labels = np.array(labels)

    # Encode labels (convert categorical labels to integers)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return keypoints, labels, label_encoder

# Prepare dataset
dataset_dir = './dataset'  # Change this to your dataset directory
keypoints, labels, label_encoder = prepare_data(dataset_dir)

# Step 2: Build a Neural Network Model for Pose Classification
model = models.Sequential()

# Input layer expects a flattened array of keypoints
model.add(layers.InputLayer(input_shape=(keypoints.shape[1],)))

# Hidden layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout to avoid overfitting
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))

# Output layer: number of classes in the dataset
model.add(layers.Dense(len(np.unique(labels)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
# Define checkpoints to save the best model during training
checkpoint = ModelCheckpoint('pose_classifier.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
model.fit(keypoints, labels, epochs=20, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Save the final model
model.save('pose_classifier_final.h5')

# Print class names
print(f"Classes: {label_encoder.classes_}")

# Step 4: Make Predictions
def predict_pose(image_path, model, label_encoder):
    image = cv2.imread(image_path)
    keypoint = extract_keypoints(image)

    if keypoint is not None:
        keypoint = np.expand_dims(keypoint, axis=0)  # Add batch dimension
        prediction = model.predict(keypoint)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_pose = label_encoder.inverse_transform([predicted_class[0]])

        return predicted_pose[0]
    else:
        print("No keypoints detected.")
        return None

# Example usage: Predict pose from an image
image_path = './sample2.jpg'  # Replace with your image path
predicted_pose = predict_pose(image_path, model, label_encoder)
print(f"Predicted Pose: {predicted_pose}")
