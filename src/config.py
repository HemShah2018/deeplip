"""
Configuration file for the lip-reading project.
Contains all hyperparameters, paths, and constants.
"""
import os

# Data paths
DATA_DIR = "data"
# To use Google Drive download, set DATA_URL to your file's shareable link
# Format: "https://drive.google.com/uc?id=YOUR_FILE_ID"
# To get file ID: Upload to Google Drive -> Right-click -> Get link -> Extract ID from URL
# Or skip download and manually place data.zip in project root, then extract it
DATA_URL = None  # Set to None to skip download, or provide Google Drive URL
DATA_ZIP_PATH = "data.zip"
ALIGNMENTS_DIR = os.path.join(DATA_DIR, "alignments")

# Video preprocessing parameters
MOUTH_REGION = {
    "top": 190,
    "bottom": 236,
    "left": 80,
    "right": 220
}
TARGET_FRAMES = 75  # Target number of frames per video clip
VIDEO_HEIGHT = MOUTH_REGION["bottom"] - MOUTH_REGION["top"]
VIDEO_WIDTH = MOUTH_REGION["right"] - MOUTH_REGION["left"]

# Vocabulary configuration
# Characters that can appear in transcripts (letters, digits, space, punctuation)
VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789 ")  # Add punctuation if needed
VOCAB_SIZE = len(VOCAB)
BLANK_TOKEN = len(VOCAB)  # CTC blank token index

# Dataset configuration
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 500
TRAIN_SIZE = 450  # Number of samples for training
MAX_TEXT_LENGTH = 40  # Maximum length of text sequences

# Model architecture
CONV3D_FILTERS = [32, 64, 128]  # Number of filters for each Conv3D layer
LSTM_UNITS = 128
NUM_LSTM_LAYERS = 2
DROPOUT_RATE = 0.5

# Training configuration
INITIAL_LEARNING_RATE = 1e-4
LR_DECAY_START_EPOCH = 30
LR_DECAY_RATE = 0.95
EPOCHS = 100
MODEL_SAVE_DIR = "models"

# Prediction configuration
DEFAULT_WEIGHTS_PATH = os.path.join(MODEL_SAVE_DIR, "weights_epoch_96.h5")

# GPU configuration
ENABLE_GPU_MEMORY_GROWTH = True

