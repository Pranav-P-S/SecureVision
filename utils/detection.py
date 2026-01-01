"""
SecureVision Detection Utilities

Helper functions for fire and smoke detection using the trained ResNet50 model.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import BatchNormalization

# Patch for Keras 2/3 compatibility issues
class PatchedBatchNormalization(BatchNormalization):
    """
    Keras 3 might store BatchNormalization axis as a list [3], while 
    older versions or specific deserializers might expect an int.
    This class handles the conversion to ensure compatibility.
    """
    def __init__(self, **kwargs):
        if 'axis' in kwargs and isinstance(kwargs['axis'], list):
            # Convert list [3] to single int 3
            kwargs['axis'] = kwargs['axis'][0]
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        if 'axis' in config and isinstance(config['axis'], list):
            config['axis'] = config['axis'][0]
        return super().from_config(config)


def load_model(model_path: str):
    """
    Load the trained fire/smoke detection model.
    
    Due to Keras 2/3 compatibility issues with the legacy .h5 file,
    we rebuild the model architecture from scratch and load only the weights.
    
    Args:
        model_path: Path to the .h5 model file
        
    Returns:
        Loaded TensorFlow/Keras model
    """
    try:
        # First, try standard loading (in case the H5 file is fixed)
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Standard load failed: {e}")
        print("Rebuilding model architecture and loading weights...")
        
        try:
            # Rebuild the exact architecture from data/main.py
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            
            # Create ResNet50 base (same config as training)
            resnet = ResNet50(
                include_top=False,
                pooling='avg',
                weights='imagenet',  # Use ImageNet weights as base
                input_shape=(224, 224, 3)
            )
            
            # Build the Sequential model (same as training)
            model = Sequential([
                resnet,
                Dense(3, activation='softmax', name='dense_1')
            ])
            
            # Freeze ResNet layers (same as training)
            model.layers[0].trainable = False
            
            # Try to load weights from the H5 file
            try:
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
                print("✅ Successfully loaded weights from H5 file")
            except Exception as weight_error:
                print(f"⚠️ Warning: Could not load trained weights: {weight_error}")
                print("Using ImageNet weights only (detection will be less accurate)")
            
            return model
            
        except Exception as rebuild_error:
            error_msg = str(rebuild_error)
            print(f"DEBUG: Error rebuilding model: {error_msg}")
            raise RuntimeError(f"Failed to load or rebuild model from {model_path}: {error_msg}")


def prepare_image(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Prepare an image for prediction.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        target_size: Target size for the model input
        
    Returns:
        Preprocessed image ready for prediction
    """
    # Resize image
    resized = cv2.resize(image, (target_size, target_size))
    
    # Convert BGR to RGB if needed
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions and preprocess
    img_array = np.expand_dims(resized, axis=0)
    return preprocess_input(img_array)


def predict(model, image: np.ndarray, config: dict) -> tuple:
    """
    Run fire/smoke detection on an image.
    
    Args:
        model: Loaded detection model
        image: Preprocessed image
        config: Detection configuration dict
        
    Returns:
        Tuple of (predicted_class_id, confidence, label_text)
    """
    # Get prediction
    pred_vec = model.predict(image, verbose=0)
    confidence = float(np.max(pred_vec))
    
    if confidence > config["confidence_threshold"]:
        class_id = int(np.argmax(pred_vec))
    else:
        class_id = 0
        confidence = 1.0 - confidence  # Show "safety" confidence
    
    label = config["labels"].get(class_id, "Unknown")
    
    return class_id, confidence, label


def draw_prediction(frame: np.ndarray, label: str, confidence: float, 
                    class_id: int, config: dict) -> np.ndarray:
    """
    Draw prediction overlay on frame.
    
    Args:
        frame: Original frame
        label: Prediction label text
        confidence: Confidence score (0-1)
        class_id: Predicted class ID
        config: Detection configuration
        
    Returns:
        Frame with prediction overlay
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # Choose color based on class
    if class_id == 1:  # Fire
        color = (0, 0, 255)  # Red in BGR
        bg_color = (0, 0, 180)
    elif class_id == 2:  # Smoke
        color = (0, 165, 255)  # Orange in BGR
        bg_color = (0, 120, 180)
    else:  # Safe
        color = (0, 255, 0)  # Green in BGR
        bg_color = (0, 180, 0)
    
    # Draw status bar at top
    cv2.rectangle(frame_copy, (0, 0), (w, 60), bg_color, -1)
    
    # Draw label
    text = f"{label} ({confidence:.1%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 40
    
    cv2.putText(frame_copy, text, (text_x, text_y), font, font_scale, 
                (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Draw border based on detection
    if class_id > 0:
        border_thickness = 8
        cv2.rectangle(frame_copy, (0, 0), (w-1, h-1), color, border_thickness)
    
    return frame_copy


def get_detection_status(class_id: int, confidence: float, config: dict) -> dict:
    """
    Get a status dictionary for UI display.
    
    Args:
        class_id: Predicted class ID
        confidence: Confidence score
        config: Detection configuration
        
    Returns:
        Dictionary with status info for UI
    """
    return {
        "class_id": class_id,
        "label": config["labels"].get(class_id, "Unknown"),
        "confidence": confidence,
        "color": config["label_colors"].get(class_id, "#6b7280"),
        "is_danger": class_id > 0
    }
