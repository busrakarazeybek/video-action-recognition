import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from model import build_model
import pandas as pd

PROCESSED_DATA_DIR = 'data/processed/'

def load_data():
    X, y = [], []
    for file in os.listdir(PROCESSED_DATA_DIR):
        if file.endswith('.npy'):
            frames = np.load(os.path.join(PROCESSED_DATA_DIR, file))
            X.append(frames)
            label = int(file.split('_')[1])  # Assuming file format: video_<label>.npy
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_model():
    X, y = load_data()
    X = np.expand_dims(X, axis=-1)  # 4D array (num_samples, frames, height, width, channels)
    input_shape = (X.shape[1], X.shape[2], X.shape[3], X.shape[4])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(input_shape)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)
    
    model.save('models/action_recognition_model.h5')
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('models/training_history.csv', index=False)

if __name__ == '__main__':
    train_model()
