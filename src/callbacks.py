"""
Training callbacks for model checkpointing, learning rate scheduling, and monitoring.
"""
import os
import tensorflow as tf
import numpy as np
from .config import (
    MODEL_SAVE_DIR, INITIAL_LEARNING_RATE,
    LR_DECAY_START_EPOCH, LR_DECAY_RATE
)


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom model checkpoint callback to save weights after each epoch.
    """
    def __init__(self, save_dir: str = MODEL_SAVE_DIR):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Save model weights after each epoch."""
        weights_path = os.path.join(
            self.save_dir,
            f"weights_epoch_{epoch+1:02d}.h5"
        )
        self.model.save_weights(weights_path)
        print(f"\nSaved weights to {weights_path}")


class LearningRateSchedule(tf.keras.callbacks.Callback):
    """
    Learning rate schedule callback.
    - First 30 epochs: constant LR (1e-4)
    - After epoch 30: exponential decay
    """
    def __init__(self, initial_lr: float = INITIAL_LEARNING_RATE,
                 decay_start_epoch: int = LR_DECAY_START_EPOCH,
                 decay_rate: float = LR_DECAY_RATE):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_start_epoch = decay_start_epoch
        self.decay_rate = decay_rate
    
    def on_epoch_begin(self, epoch: int, logs: dict = None):
        """Update learning rate at the beginning of each epoch."""
        if epoch < self.decay_start_epoch:
            lr = self.initial_lr
        else:
            # Exponential decay
            decay_epochs = epoch - self.decay_start_epoch + 1
            lr = self.initial_lr * (self.decay_rate ** decay_epochs)
        
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"\nEpoch {epoch+1}: Learning rate = {lr:.6f}")


class ProduceExample(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor predictions during training.
    Prints ground truth and predicted text at the end of each epoch.
    """
    def __init__(self, val_dataset: tf.data.Dataset, num_to_char: tf.keras.layers.StringLookup):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_to_char = num_to_char
    
    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Generate and print predictions on a validation batch."""
        # Get one batch from validation set
        for batch_videos, batch_labels in self.val_dataset.take(1):
            # Get predictions
            predictions = self.model.predict(batch_videos, verbose=0)
            
            # Decode predictions using CTC
            input_length = np.ones(predictions.shape[0]) * predictions.shape[1]
            
            # Use CTC decode (greedy)
            decoded, _ = tf.keras.backend.ctc_decode(
                predictions,
                input_length,
                greedy=True
            )
            
            # Convert predictions to characters
            decoded = decoded[0]  # Get the first (and only) result
            
            print("\n" + "="*60)
            print(f"Epoch {epoch+1} - Example Predictions:")
            print("="*60)
            
            for i in range(min(2, len(batch_labels))):  # Show up to 2 examples
                # Ground truth
                gt_labels = batch_labels[i].numpy()
                gt_labels = gt_labels[gt_labels != 0]  # Remove padding
                if len(gt_labels) > 0:
                    gt_tensor = tf.constant(gt_labels)
                    gt_chars = self.num_to_char(gt_tensor).numpy()
                    gt_text = b''.join(gt_chars).decode('utf-8')
                else:
                    gt_text = ""
                
                # Prediction
                pred_labels = decoded[i].numpy()
                pred_labels = pred_labels[pred_labels != -1]  # Remove CTC blank (-1)
                pred_labels = pred_labels[pred_labels != 0]  # Remove padding
                if len(pred_labels) > 0:
                    pred_tensor = tf.constant(pred_labels)
                    pred_chars = self.num_to_char(pred_tensor).numpy()
                    pred_text = b''.join(pred_chars).decode('utf-8')
                else:
                    pred_text = ""
                
                print(f"\nExample {i+1}:")
                print(f"  Ground Truth: {gt_text}")
                print(f"  Predicted:    {pred_text}")
            
            print("="*60 + "\n")
            break  # Only process one batch

