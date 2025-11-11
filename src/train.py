"""
Training script for the lip-reading model.
"""
import os
import tensorflow as tf
import argparse
from .config import (
    DATA_DIR, BATCH_SIZE, EPOCHS, TRAIN_SIZE,
    INITIAL_LEARNING_RATE, MODEL_SAVE_DIR
)
from .dataset import build_vocab_lookup, create_dataset, prepare_dataset, split_dataset
from .model import build_model, print_model_summary
from .losses import ctc_loss_fn
from .callbacks import ModelCheckpoint, LearningRateSchedule, ProduceExample


def setup_gpu():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found. Using CPU.")


def main(video_pattern: str = None, epochs: int = EPOCHS):
    """
    Main training function.
    
    Args:
        video_pattern: Glob pattern for video files (e.g., "data/S1/*.mpg")
        epochs: Number of training epochs
    """
    # Setup GPU
    setup_gpu()
    
    # Default video pattern if not provided
    if video_pattern is None:
        # Try to find video files in data directory (try MP4 first, then MPG)
        video_pattern = os.path.join(DATA_DIR, "S1", "*.mp4")
        if not any(Path(DATA_DIR).glob("S*/*.mp4")):
            video_pattern = os.path.join(DATA_DIR, "S1", "*.mpg")
        if not os.path.exists(os.path.dirname(video_pattern)):
            # Try alternative patterns
            video_pattern = os.path.join(DATA_DIR, "*", "*.mp4")
            if not any(Path(DATA_DIR).glob("*/*.mp4")):
                video_pattern = os.path.join(DATA_DIR, "*", "*.mpg")
    
    print(f"Using video pattern: {video_pattern}")
    
    # Build vocabulary lookup layers
    print("Building vocabulary lookup...")
    char_to_num, num_to_char = build_vocab_lookup()
    print(f"Vocabulary size: {len(char_to_num.get_vocabulary())}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(video_pattern, char_to_num, shuffle=True)
    
    # Prepare dataset (padding, batching)
    dataset = prepare_dataset(dataset)
    
    # Split into train and validation
    # Adjust train_size based on actual dataset size
    dataset_list = list(dataset)
    total_samples = len(dataset_list)
    train_size = min(TRAIN_SIZE, int(total_samples * 0.8))  # Use 80% for training
    
    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size)
    
    # Count samples (approximate)
    train_size_approx = sum(1 for _ in train_data)
    val_size_approx = sum(1 for _ in val_data)
    print(f"Training samples: ~{train_size_approx * BATCH_SIZE}")
    print(f"Validation samples: ~{val_size_approx * BATCH_SIZE}")
    
    # If no validation data, use training data for validation (small dataset)
    if val_size_approx == 0:
        print("Warning: No validation data. Using training data for validation.")
        val_data = train_data.take(5)  # Use small subset for validation
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    print_model_summary(model)
    
    # Compile model
    print("\nCompiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=ctc_loss_fn,
        metrics=[]  # CTC doesn't have standard metrics
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(save_dir=MODEL_SAVE_DIR),
        LearningRateSchedule(
            initial_lr=INITIAL_LEARNING_RATE,
            decay_start_epoch=30,
            decay_rate=0.95
        ),
        ProduceExample(val_dataset=val_data, num_to_char=num_to_char)
    ]
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    print("="*60)
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed!")
    print(f"Model weights saved in {MODEL_SAVE_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lip-reading model")
    parser.add_argument(
        "--video_pattern",
        type=str,
        default=None,
        help="Glob pattern for video files (e.g., 'data/S1/*.mpg')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    
    args = parser.parse_args()
    main(video_pattern=args.video_pattern, epochs=args.epochs)

