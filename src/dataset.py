"""
Dataset utilities for building tf.data pipelines.
Handles vocabulary setup and dataset creation.
"""
import tensorflow as tf
from typing import Tuple
from .config import (
    VOCAB, VOCAB_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE,
    TARGET_FRAMES, MAX_TEXT_LENGTH, TRAIN_SIZE
)
from .data import load_data_tf


def build_vocab_lookup():
    """
    Build character-to-number and number-to-character lookup layers.
    
    Returns:
        Tuple of (char_to_num, num_to_char) StringLookup layers
    """
    # Create char_to_num lookup
    char_to_num = tf.keras.layers.StringLookup(
        vocabulary=VOCAB,
        mask_token="",
        oov_token=""
    )
    
    # Create num_to_char lookup (inverse)
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=VOCAB,
        mask_token="",
        oov_token="",
        invert=True
    )
    
    return char_to_num, num_to_char


def create_dataset(video_pattern: str, char_to_num: tf.keras.layers.StringLookup, 
                   shuffle: bool = True) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from video files.
    
    Args:
        video_pattern: Glob pattern for video files (e.g., "data/S1/*.mpg")
        char_to_num: StringLookup layer for character encoding
        shuffle: Whether to shuffle the dataset
        
    Returns:
        tf.data.Dataset yielding (video, alignment) tuples
    """
    # List all video files
    dataset = tf.data.Dataset.list_files(video_pattern, shuffle=shuffle)
    
    # Shuffle file paths
    if shuffle:
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    
    # Map paths to (video, alignment) pairs
    dataset = dataset.map(
        lambda path: load_data_tf(path, char_to_num),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset


def pad_video(video: tf.Tensor, target_frames: int = TARGET_FRAMES) -> tf.Tensor:
    """
    Pad or truncate video to target number of frames.
    
    Args:
        video: Video tensor of shape [T, H, W, 1]
        target_frames: Target number of frames
        
    Returns:
        Padded/truncated video tensor of shape [target_frames, H, W, 1]
    """
    current_frames = tf.shape(video)[0]
    
    # Truncate if too long
    video = video[:target_frames]
    
    # Pad if too short
    padding_needed = target_frames - tf.shape(video)[0]
    padding = tf.zeros([padding_needed, tf.shape(video)[1], tf.shape(video)[2], 1], 
                       dtype=video.dtype)
    video = tf.concat([video, padding], axis=0)
    
    return video


def pad_sequence(sequence: tf.Tensor, max_length: int = MAX_TEXT_LENGTH, 
                 pad_value: int = 0) -> tf.Tensor:
    """
    Pad or truncate sequence to max_length.
    
    Args:
        sequence: Sequence tensor of shape [L]
        max_length: Maximum sequence length
        pad_value: Value to use for padding
        
    Returns:
        Padded/truncated sequence tensor of shape [max_length]
    """
    current_length = tf.shape(sequence)[0]
    
    # Truncate if too long
    sequence = sequence[:max_length]
    
    # Pad if too short
    padding_needed = max_length - tf.shape(sequence)[0]
    padding = tf.fill([padding_needed], pad_value)
    sequence = tf.concat([sequence, padding], axis=0)
    
    return sequence


def prepare_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Prepare dataset with padding and batching.
    
    Args:
        dataset: Raw dataset yielding (video, alignment) tuples
        
    Returns:
        Batched and prefetched dataset
    """
    # Pad videos and alignments
    def pad_data(video, alignment):
        video = pad_video(video, TARGET_FRAMES)
        alignment = pad_sequence(alignment, MAX_TEXT_LENGTH, pad_value=0)
        return video, alignment
    
    dataset = dataset.map(pad_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch with padding
    dataset = dataset.padded_batch(
        batch_size=BATCH_SIZE,
        padded_shapes=(
            [TARGET_FRAMES, None, None, 1],  # Video shape
            [MAX_TEXT_LENGTH]  # Alignment shape
        ),
        padding_values=(0.0, 0)  # Padding values
    )
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def split_dataset(dataset: tf.data.Dataset, train_size: int = TRAIN_SIZE) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: Full dataset
        train_size: Number of samples for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size)
    
    return train_data, val_data

