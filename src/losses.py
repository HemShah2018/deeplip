"""
Custom loss functions for CTC training.
"""
import tensorflow as tf
from .config import TARGET_FRAMES


def ctc_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    CTC loss function for sequence-to-sequence learning.
    
    Args:
        y_true: Integer labels of shape [batch, max_text_length]
        y_pred: Model predictions of shape [batch, 75, vocab_size+1]
        
    Returns:
        Scalar CTC loss value
    """
    batch_size = tf.shape(y_true)[0]
    max_label_len = tf.shape(y_true)[1]
    
    # Compute label lengths
    label_length = tf.reduce_sum(
        tf.cast(tf.not_equal(y_true, 0), tf.int32),
        axis=1
    )
    label_length = tf.maximum(label_length, 1)
    
    # Input length
    input_length = tf.fill([batch_size], TARGET_FRAMES)
    
    # Convert dense labels to sparse format using tf operations
    # Create mask for non-zero values
    mask = tf.not_equal(y_true, 0)
    
    # Get indices where mask is True
    batch_indices = tf.where(mask)[:, 0]
    seq_indices = tf.where(mask)[:, 1]
    values = tf.gather_nd(y_true, tf.where(mask))
    
    # Create sparse tensor
    indices = tf.stack([batch_indices, seq_indices], axis=1)
    sparse_labels = tf.SparseTensor(
        indices=indices,
        values=tf.cast(values, tf.int32),
        dense_shape=[batch_size, max_label_len]
    )
    
    # Ensure sparse tensor is properly ordered
    sparse_labels = tf.sparse.reorder(sparse_labels)
    
    # Convert predictions to logits
    y_pred_log = tf.math.log(tf.maximum(y_pred, 1e-8))
    
    # Use tf.nn.ctc_loss
    loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=y_pred_log,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=False,
        blank_index=-1  # Use default blank index
    )
    
    return tf.reduce_mean(loss)
