"""
Prediction script for lip-reading inference.
"""
import os
import numpy as np
import tensorflow as tf
import argparse
from .config import (
    TARGET_FRAMES, MAX_TEXT_LENGTH, DEFAULT_WEIGHTS_PATH,
    VIDEO_HEIGHT, VIDEO_WIDTH
)
from .data import load_video
from .dataset import build_vocab_lookup
from .model import build_model


def pad_video(video: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Pad or truncate video to target number of frames.
    
    Args:
        video: Video array of shape [T, H, W, 1]
        target_frames: Target number of frames
        
    Returns:
        Padded/truncated video array of shape [target_frames, H, W, 1]
    """
    current_frames = video.shape[0]
    
    if current_frames > target_frames:
        # Truncate
        video = video[:target_frames]
    elif current_frames < target_frames:
        # Pad
        padding = np.zeros(
            (target_frames - current_frames, video.shape[1], video.shape[2], 1),
            dtype=video.dtype
        )
        video = np.concatenate([video, padding], axis=0)
    
    return video


def decode_predictions(predictions: np.ndarray, num_to_char: tf.keras.layers.StringLookup) -> str:
    """
    Decode CTC predictions to text string.
    
    Args:
        predictions: Model predictions of shape [batch, 75, vocab_size+1]
        num_to_char: StringLookup layer for converting IDs to characters
        
    Returns:
        Decoded text string
    """
    # Use CTC decode (greedy)
    input_length = np.ones(predictions.shape[0]) * predictions.shape[1]
    
    decoded, _ = tf.keras.backend.ctc_decode(
        predictions,
        input_length,
        greedy=True
    )
    
    # Get first (and only) result
    decoded = decoded[0].numpy()[0]  # Shape: [sequence_length]
    
    # Remove CTC blank tokens (-1) and padding
    decoded = decoded[decoded != -1]
    decoded = decoded[decoded != 0]
    
    if len(decoded) == 0:
        return ""
    
    # Convert to characters
    decoded_tensor = tf.constant(decoded)
    char_tensors = num_to_char(decoded_tensor)
    
    # Convert tensor to string
    if isinstance(char_tensors, tf.Tensor):
        chars = [char.decode('utf-8') for char in char_tensors.numpy()]
    else:
        chars = [char.decode('utf-8') if isinstance(char, bytes) else str(char) 
                 for char in char_tensors]
    
    # Join characters into string
    text = ''.join(chars)
    
    return text


def predict_clip(video_path: str, model: tf.keras.Model, 
                 num_to_char: tf.keras.layers.StringLookup) -> str:
    """
    Predict text from a video clip.
    
    Args:
        video_path: Path to video file
        model: Trained Keras model
        num_to_char: StringLookup layer for decoding
        
    Returns:
        Predicted text string
    """
    # Load and preprocess video
    video = load_video(video_path)
    
    # Pad/truncate to target frames
    video = pad_video(video, TARGET_FRAMES)
    
    # Add batch dimension
    video = np.expand_dims(video, axis=0)  # Shape: [1, 75, H, W, 1]
    
    # Ensure correct shape
    if video.shape[1] != TARGET_FRAMES:
        video = pad_video(video[0], TARGET_FRAMES)
        video = np.expand_dims(video, axis=0)
    
    # Predict
    predictions = model.predict(video, verbose=0)
    
    # Decode predictions
    text = decode_predictions(predictions, num_to_char)
    
    return text


def load_model(weights_path: str = DEFAULT_WEIGHTS_PATH) -> tf.keras.Model:
    """
    Load trained model with weights.
    
    Args:
        weights_path: Path to model weights file
        
    Returns:
        Loaded and compiled model
    """
    print(f"Loading model from {weights_path}...")
    
    # Build model
    model = build_model()
    
    # Load weights
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Weights loaded successfully.")
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Model initialized with random weights.")
    
    # Compile model (needed for prediction)
    import tensorflow as tf
    from .losses import ctc_loss_fn
    from .config import INITIAL_LEARNING_RATE
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss=ctc_loss_fn
    )
    
    return model


def main(video_path: str, weights_path: str = DEFAULT_WEIGHTS_PATH):
    """
    Main prediction function.
    
    Args:
        video_path: Path to video file for prediction
        weights_path: Path to model weights file
    """
    # Build vocabulary lookup
    _, num_to_char = build_vocab_lookup()
    
    # Load model
    model = load_model(weights_path)
    
    # Predict
    print(f"\nPredicting text from video: {video_path}")
    predicted_text = predict_clip(video_path, model, num_to_char)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT:")
    print("="*60)
    print(f"Predicted text: {predicted_text}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict text from video using trained model")
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file (.mpg)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Path to model weights file (default: {DEFAULT_WEIGHTS_PATH})"
    )
    
    args = parser.parse_args()
    main(video_path=args.video_path, weights_path=args.weights)

