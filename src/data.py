"""
Data loading and preprocessing utilities.
Handles video loading, alignment parsing, and data download.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import gdown
import zipfile
from typing import Tuple
from .config import DATA_DIR, DATA_URL, DATA_ZIP_PATH, ALIGNMENTS_DIR, MOUTH_REGION


def download_and_extract_data():
    """
    Download data from Google Drive and extract to data directory.
    
    Steps:
    1. Downloads data.zip (~423 MB) from Google Drive using gdown (if URL provided)
    2. Extracts the zip file to the data/ directory
    3. The extracted folder contains videos and alignment files
    
    If DATA_URL is None, it will skip download and only extract if data.zip exists locally.
    """
    # Check if we should download or use local file
    if DATA_URL is None:
        if os.path.exists(DATA_ZIP_PATH):
            print(f"DATA_URL is None. Using local {DATA_ZIP_PATH} file.")
        else:
            print(f"DATA_URL is None and {DATA_ZIP_PATH} not found.")
            print("Please either:")
            print("  1. Set DATA_URL in src/config.py with your Google Drive file ID")
            print("  2. Place data.zip in the project root directory")
            return
    else:
        # Step 1: Download the zip file using gdown
        print(f"Downloading data from {DATA_URL}...")
        print(f"Output file: {DATA_ZIP_PATH} (~423 MB)")
        try:
            gdown.download(DATA_URL, DATA_ZIP_PATH, quiet=False)
        except Exception as e:
            print(f"Download failed: {e}")
            print("Make sure the Google Drive file is set to 'Anyone with the link can view'")
            return
    
    # Step 2: Extract to data directory using zipfile
    # (gdown doesn't have extract_all, so we use Python's zipfile module)
    if not os.path.exists(DATA_ZIP_PATH):
        print(f"Error: {DATA_ZIP_PATH} not found. Cannot extract.")
        return
    
    print(f"Extracting {DATA_ZIP_PATH} to {DATA_DIR}...")
    try:
        with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"Data extracted to {DATA_DIR}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return
    
    # Optionally remove zip file after extraction
    # os.remove(DATA_ZIP_PATH)


def load_video(video_path: str) -> np.ndarray:
    """
    Load video file, extract mouth region, and preprocess frames.
    
    Args:
        video_path: Path to the video file (.mpg, .mp4, or other formats supported by OpenCV)
        
    Returns:
        Preprocessed video tensor of shape [T, H, W, 1]
    """
    # Read video using OpenCV (supports .mpg, .mp4, .avi, etc.)
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Crop mouth region (using static coordinates)
        mouth_crop = gray[
            MOUTH_REGION["top"]:MOUTH_REGION["bottom"],
            MOUTH_REGION["left"]:MOUTH_REGION["right"]
        ]
        
        frames.append(mouth_crop)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")
    
    # Stack frames into array
    video_array = np.stack(frames, axis=0)  # Shape: [T, H, W]
    
    # Add channel dimension
    video_array = np.expand_dims(video_array, axis=-1)  # Shape: [T, H, W, 1]
    
    # Standardize: compute mean and std, then normalize
    mean = np.mean(video_array)
    std = np.std(video_array)
    
    if std > 0:
        video_array = (video_array - mean) / std
    else:
        video_array = video_array - mean
    
    # Cast to float32
    video_array = video_array.astype(np.float32)
    
    return video_array


def load_alignments(align_path: str, char_to_num: tf.keras.layers.StringLookup) -> np.ndarray:
    """
    Load and parse alignment file, convert to integer sequence.
    
    Args:
        align_path: Path to the .align file
        char_to_num: StringLookup layer for character to integer conversion
        
    Returns:
        Integer sequence of shape [text_length]
    """
    with open(align_path, 'r') as f:
        lines = f.readlines()
    
    # Parse alignment file
    # Format: start_time end_time token
    tokens = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            token = parts[2].lower()  # Get token and convert to lowercase
            if token != "silence":
                tokens.append(token)
    
    # Join tokens into sentence
    sentence = " ".join(tokens)
    
    # Convert sentence to character sequence
    char_sequence = list(sentence)
    
    # Convert characters to integers using StringLookup
    # StringLookup expects batched input, so we pass the entire sequence
    if len(char_sequence) > 0:
        char_tensor = char_to_num(char_sequence)
        char_ids = char_tensor.numpy().astype(np.int32)
    else:
        char_ids = np.array([], dtype=np.int32)
    
    return char_ids


def load_data(video_path: str, char_to_num: tf.keras.layers.StringLookup) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load video and corresponding alignment data.
    
    Args:
        video_path: Path to video file
        char_to_num: StringLookup layer for character encoding
        
    Returns:
        Tuple of (video_tensor, alignment_tensor)
    """
    # Derive alignment path from video path
    # Convert data/S1/XXXXXX.mpg -> data/alignments/S1/XXXXXX.align
    video_dir = os.path.dirname(video_path)
    video_filename = os.path.basename(video_path)
    video_basename = os.path.splitext(video_filename)[0]
    
    # Extract speaker directory (e.g., S1) from video path
    speaker_dir = os.path.basename(video_dir)
    
    align_path = os.path.join(ALIGNMENTS_DIR, speaker_dir, f"{video_basename}.align")
    
    # Load video and alignments
    video = load_video(video_path)
    alignment = load_alignments(align_path, char_to_num)
    
    return video, alignment


def load_data_tf(video_path: tf.Tensor, char_to_num: tf.keras.layers.StringLookup) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow wrapper for load_data to use in tf.data pipeline.
    
    Args:
        video_path: Tensor containing video path string
        char_to_num: StringLookup layer for character encoding
        
    Returns:
        Tuple of (video_tensor, alignment_tensor)
    """
    def _load_py(video_path_str):
        video_path_str = video_path_str.numpy().decode('utf-8')
        video, alignment = load_data(video_path_str, char_to_num)
        return video, alignment
    
    video, alignment = tf.py_function(
        func=_load_py,
        inp=[video_path],
        Tout=[tf.float32, tf.int32]
    )
    
    # Set shapes for TensorFlow
    video.set_shape([None, MOUTH_REGION["bottom"] - MOUTH_REGION["top"], 
                     MOUTH_REGION["right"] - MOUTH_REGION["left"], 1])
    alignment.set_shape([None])
    
    return video, alignment

