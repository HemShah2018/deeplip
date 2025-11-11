"""
Visualization utilities for inspecting preprocessed video clips.
"""
import numpy as np
import imageio
from .data import load_video
from .config import TARGET_FRAMES


def save_video_animation(video: np.ndarray, output_path: str = "animation.gif"):
    """
    Save a preprocessed video clip as an animated GIF for inspection.
    
    Args:
        video: Video array of shape [T, H, W, 1] or [T, H, W]
        output_path: Path to save the GIF file
    """
    # Handle different input shapes
    if len(video.shape) == 4:
        # Remove channel dimension if present
        video = video.squeeze(axis=-1)
    
    # Denormalize for visualization (assuming video was standardized)
    # Clip values to [0, 255] range
    video_vis = video.copy()
    
    # Normalize to [0, 1] range
    v_min = video_vis.min()
    v_max = video_vis.max()
    if v_max > v_min:
        video_vis = (video_vis - v_min) / (v_max - v_min)
    else:
        video_vis = np.zeros_like(video_vis)
    
    # Convert to uint8
    video_vis = (video_vis * 255).astype(np.uint8)
    
    # Save as GIF
    imageio.mimsave(output_path, video_vis, fps=10)
    print(f"Saved animation to {output_path}")


def visualize_preprocessed_clip(video_path: str, output_path: str = "animation.gif"):
    """
    Load a video, preprocess it, and save as an animated GIF.
    
    Args:
        video_path: Path to video file
        output_path: Path to save the GIF file
    """
    # Load and preprocess video
    video = load_video(video_path)
    
    # Pad/truncate to target frames
    current_frames = video.shape[0]
    if current_frames > TARGET_FRAMES:
        video = video[:TARGET_FRAMES]
    elif current_frames < TARGET_FRAMES:
        padding = np.zeros(
            (TARGET_FRAMES - current_frames, video.shape[1], video.shape[2], 1),
            dtype=video.dtype
        )
        video = np.concatenate([video, padding], axis=0)
    
    # Save as animation
    save_video_animation(video, output_path)

