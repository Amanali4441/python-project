import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Initialize mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the pre-trained pose classification model (replace with your model path)
model = tf.keras.models.load_model('./pose_classifier_final.h5')

# Function to extract pose landmarks from an image
def extract_pose_landmarks(image):
    # Convert the image to RGB (as MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Extract x, y, z coordinates of each landmark
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])  # x, y, z coordinates
        return np.array(landmarks).flatten()  # Flatten to a 1D array
    else:
        return None


# Load pose class names from dataset folder names (assumes folder names represent pose classes)
pose_class_names = sorted(os.listdir('./dataset'))  # Replace with the actual path to your dataset
# Filter out any non-directory items in case there are files
pose_class_names = [name for name in pose_class_names if os.path.isdir(os.path.join('./dataset', name))]

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Extract pose landmarks
    pose_landmarks = extract_pose_landmarks(frame)
    
    if pose_landmarks is not None:
        # Prepare the data for classification (reshape if necessary for your model)
        pose_landmarks = np.expand_dims(pose_landmarks, axis=0)  # Adding batch dimension
        
        # Predict the pose class using the trained model
        pose_class = model.predict(pose_landmarks)
        pose_class_label = np.argmax(pose_class, axis=1)  # Assuming it's a classification task
        
        # Get the pose name using the class label
        predicted_pose_name = pose_class_names[pose_class_label[0]]
        
        # Display the predicted pose class name
        cv2.putText(frame, f"Pose: {predicted_pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with pose label
    cv2.imshow("Pose Classification", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()