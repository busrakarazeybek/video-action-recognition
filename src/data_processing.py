import os
import cv2
import numpy as np

RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'

def preprocess_video(video_path):
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
    return frames

def save_preprocessed_video(frames, save_path):
    np.save(save_path, frames)

def process_all_videos():
    for video_file in os.listdir(RAW_DATA_DIR):
        video_path = os.path.join(RAW_DATA_DIR, video_file)
        frames = preprocess_video(video_path)
        save_path = os.path.join(PROCESSED_DATA_DIR, video_file.split('.')[0] + '.npy')
        save_preprocessed_video(frames, save_path)

if __name__ == '__main__':
    process_all_videos()
