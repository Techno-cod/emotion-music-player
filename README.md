# Real-Time Facial Emotion Recognition System

A computer vision pipeline that detects facial emotions via webcam and plays mood-matching music in real time.

## What it does
- Captures live webcam feed using OpenCV
- Runs two emotion classifiers simultaneously:
  - **DeepFace** (VGG-Face backbone) for music selection
  - **Custom PyTorch CNN** trained on FER-2013 for real-time inference
- Plays mood-matched music based on detected emotion
- Displays both model predictions on screen simultaneously

## PyTorch Model
- 3-layer CNN trained from scratch on FER-2013 dataset (35,000 facial images)
- Architecture: Conv2d → MaxPool → Conv2d → MaxPool → Conv2d → MaxPool → FC → FC
- Test accuracy: **58.48%** across 7 emotion classes
- Runs on Apple M1 MPS backend for CPU-efficient inference

## Tech Stack
Python, PyTorch, OpenCV, DeepFace, Pygame

## Detection accuracy
- PyTorch model: 58.48% on FER-2013 test set
- Optimised inference loop — 4-second detection interval to reduce CPU load

## How to run
```bash
pip install torch torchvision opencv-python deepface pygame
python3 emotion_model.py   # train the PyTorch model first
python3 main.py            # run the live pipeline
```

## Key engineering decisions
- PyTorch model predicts on full frame; DeepFace crops face region first — trade-off between speed and accuracy noted
- Duplicate-play prevention logic avoids replaying the same song
- Modular design: model training (emotion_model.py) separated from inference pipeline (main.py)
