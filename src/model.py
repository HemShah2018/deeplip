"""
Neural network model definition.
Conv3D + BiLSTM + CTC architecture for lip reading.
"""
import tensorflow as tf
from .config import (
    TARGET_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH,
    CONV3D_FILTERS, LSTM_UNITS, NUM_LSTM_LAYERS,
    DROPOUT_RATE, VOCAB_SIZE, BLANK_TOKEN
)


def build_model(input_shape: tuple = None) -> tf.keras.Model:
    """
    Build the lip-reading model architecture.
    
    Architecture:
    - Input: [batch, 75, H, W, 1]
    - Conv3D blocks with MaxPool3D
    - TimeDistributed Flatten
    - Bidirectional LSTM layers
    - Dropout
    - Dense output with Softmax
    
    Args:
        input_shape: Input shape tuple (time, height, width, channels)
                     If None, uses default from config
        
    Returns:
        Uncompiled Keras model
    """
    if input_shape is None:
        input_shape = (TARGET_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 1)
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="video_input")
    x = inputs
    
    # Conv3D blocks
    for i, filters in enumerate(CONV3D_FILTERS):
        x = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=(3, 3, 3),
            padding="same",
            activation="relu",
            name=f"conv3d_{i+1}"
        )(x)
        
        # MaxPool3D - reduce spatial dimensions but preserve temporal
        # Use (1, 2, 2) pooling to keep time dimension intact
        x = tf.keras.layers.MaxPool3D(
            pool_size=(1, 2, 2),
            padding="same",
            name=f"maxpool3d_{i+1}"
        )(x)
    
    # TimeDistributed Flatten to collapse spatial dimensions
    # This preserves the temporal dimension (75 time steps)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(),
        name="time_distributed_flatten"
    )(x)
    
    # Bidirectional LSTM layers
    for i in range(NUM_LSTM_LAYERS):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                LSTM_UNITS,
                return_sequences=True,
                name=f"lstm_{i+1}"
            ),
            name=f"bidirectional_lstm_{i+1}"
        )(x)
    
    # Dropout for regularization
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    
    # Dense output layer with Softmax
    # Output shape: [batch, 75, vocab_size + 1]
    # The +1 is for the CTC blank token
    outputs = tf.keras.layers.Dense(
        VOCAB_SIZE + 1,  # vocab_size + blank token
        activation="softmax",
        name="output"
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lip_reading_model")
    
    return model


def print_model_summary(model: tf.keras.Model):
    """
    Print model summary and parameter count.
    
    Args:
        model: Keras model
    """
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Expected parameters: ~8-9 million")

