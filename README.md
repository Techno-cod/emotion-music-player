# Real-Time Facial Emotion Recognition System

A computer vision pipeline that detects facial emotions via webcam and plays mood-matching music in real time. Includes a custom PyTorch CNN trained on FER-2013, with full evaluation pipeline and confusion matrix analysis.

## Results

| Emotion | Accuracy |
|---------|----------|
| happy | 80.1% |
| surprise | 71.2% |
| neutral | 60.7% |
| angry | 52.5% |
| disgust | 45.0% |
| sad | 40.5% |
| fear | 37.0% |
| **Overall** | **58.48%** |

Happy and surprise perform strongest due to distinct facial muscle patterns. Fear and sad are hardest — consistent with human difficulty distinguishing these emotions, and a known challenge in the FER-2013 dataset.

## Architecture

- 3× Conv2d layers (1→32→64→128 filters) + MaxPool2d + Dropout(0.5)
- Trained on FER-2013 (35,000 facial images, 7 emotion classes)
- Apple M1 MPS backend for GPU-accelerated training

## Two simultaneous classifiers

The live pipeline runs two models side by side:
- Custom PyTorch CNN (trained from scratch on FER-2013)
- DeepFace (VGG-Face backbone) for music selection

Both predictions are displayed on screen simultaneously.

## Evaluation pipeline (`evaluate.py`)

- Per-class accuracy breakdown across all 7 emotions
- Confusion matrix with matplotlib visualisation
- Results exported to `results.json`

## Tech stack

Python, PyTorch, OpenCV, DeepFace, Pygame, scikit-learn, matplotlib

## How to run

```bash
pip install torch torchvision opencv-python deepface pygame scikit-learn matplotlib
python3 emotion_model.py   # train PyTorch model
python3 evaluate.py        # run evaluation + generate confusion matrix
python3 main.py            # run live pipeline
```
