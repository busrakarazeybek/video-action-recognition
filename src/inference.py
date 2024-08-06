import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = 'models/action_recognition_model.h5'
VIDEO_PATH = 'path_to_new_video.mp4'
OUTPUT_PATH = 'output_video_with_predictions.mp4'

def preprocess_video_for_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    
    cap.release()
    frames = np.array(frames)
    frames = frames / 255.0
    return frames

def make_inference(video_frames, model):
    video_frames = np.expand_dims(video_frames, axis=-1)
    predictions = model.predict(video_frames)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

def visualize_predictions(video_path, predicted_labels, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label = predicted_labels[idx]
        cv2.putText(frame, f'Action: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        idx += 1
    
    cap.release()
    out.release()

if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    video_frames = preprocess_video_for_inference(VIDEO_PATH)
    predicted_labels = make_inference(video_frames, model)
    visualize_predictions(VIDEO_PATH, predicted_labels, OUTPUT_PATH)
