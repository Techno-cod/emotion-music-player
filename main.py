import cv2
import pygame
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
from deepface import DeepFace

# --- PyTorch Emotion Model ---
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 6 * 6, 512)
        self.fc2   = nn.Linear(512, num_classes)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load PyTorch model
device = torch.device("mps")
classes = torch.load('emotion_classes.pth')
pytorch_model = EmotionCNN(num_classes=len(classes)).to(device)
pytorch_model.load_state_dict(torch.load('emotion_cnn.pth', map_location=device))
pytorch_model.eval()
print(f"PyTorch model loaded — classes: {classes}")

def predict_emotion_pytorch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = (tensor / 255.0 - 0.5) / 0.5
    tensor = tensor.to(device)
    with torch.no_grad():
        output = pytorch_model(tensor)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

# --- Setup ---
pygame.mixer.init()
SONGS_DIR = "songs"
EMOTIONS = ["happy", "sad", "angry", "neutral"]

def get_song(emotion):
    folder = os.path.join(SONGS_DIR, emotion)
    if not os.path.exists(folder):
        return None
    songs = [f for f in os.listdir(folder) if f.endswith(".mp3")]
    return os.path.join(folder, random.choice(songs)) if songs else None

def play_song(emotion):
    song = get_song(emotion)
    if song:
        pygame.mixer.music.load(song)
        pygame.mixer.music.play(-1)
        print(f"Playing: {song}")

# --- Main ---
cap = cv2.VideoCapture(0)
current_emotion = None
last_check = time.time()
CHECK_INTERVAL = 4

print("Starting Emotion Music Player... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    if time.time() - last_check > CHECK_INTERVAL:
        try:
            # DeepFace prediction
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            deepface_emotion = result[0]["dominant_emotion"]

            # PyTorch prediction
            pytorch_emotion = predict_emotion_pytorch(frame)

            print(f"DeepFace: {deepface_emotion} | PyTorch: {pytorch_emotion}")

            # Use DeepFace for music (more accurate), PyTorch shown on screen
            emotion_map = {
                "happy": "happy", "surprise": "happy",
                "sad": "sad", "fear": "sad", "disgust": "sad",
                "angry": "angry",
                "neutral": "neutral"
            }
            mapped = emotion_map.get(deepface_emotion, "neutral")

            if mapped != current_emotion:
                current_emotion = mapped
                pygame.mixer.music.stop()
                play_song(current_emotion)

            last_check = time.time()

        except Exception as e:
            print("Detection error:", e)

    # Display both predictions on screen
    cv2.putText(display, f"DeepFace: {current_emotion or 'detecting...'}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2)
    cv2.putText(display, f"PyTorch:  {predict_emotion_pytorch(frame)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
    cv2.imshow("Emotion Music Player", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()