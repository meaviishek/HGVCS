#!/usr/bin/env python3
"""
Gesture Model Training Script

This script trains a deep learning model for hand gesture recognition
using collected training data.

Usage:
    python scripts/train_gesture_model.py --data data/training/gestures --output models/gesture_model.h5
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_data(data_dir: str, sequence_length: int = 30):
    """
    Load gesture training data from directory.
    
    Expected structure:
        data_dir/
            gesture_name_1/
                sequence_001.npy
                sequence_002.npy
                ...
            gesture_name_2/
                ...
    
    Args:
        data_dir: Path to training data directory
        sequence_length: Expected sequence length
    
    Returns:
        X: Array of shape (n_samples, sequence_length, n_features)
        y: Array of labels
        gesture_names: List of gesture class names
    """
    print(f"Loading data from: {data_dir}")
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    sequences = []
    labels = []
    gesture_names = []
    
    # Iterate through gesture folders
    for gesture_idx, gesture_name in enumerate(sorted(os.listdir(data_dir))):
        gesture_path = data_dir / gesture_name
        
        if not gesture_path.is_dir():
            continue
        
        gesture_names.append(gesture_name)
        print(f"  Loading '{gesture_name}'...", end=" ")
        
        gesture_count = 0
        for sequence_file in gesture_path.glob("*.npy"):
            try:
                sequence = np.load(sequence_file)
                
                # Validate sequence shape
                if len(sequence) != sequence_length:
                    print(f"\n    Warning: Skipping {sequence_file} - wrong length ({len(sequence)} != {sequence_length})")
                    continue
                
                sequences.append(sequence)
                labels.append(gesture_idx)
                gesture_count += 1
                
            except Exception as e:
                print(f"\n    Error loading {sequence_file}: {e}")
        
        print(f"({gesture_count} samples)")
    
    if not sequences:
        raise ValueError("No training data found!")
    
    X = np.array(sequences)
    y = np.array(labels)
    
    print(f"\nLoaded {len(X)} samples across {len(gesture_names)} gesture classes")
    print(f"Feature shape: {X.shape}")
    
    return X, y, gesture_names

def build_model(input_shape: tuple, num_classes: int, model_type: str = "cnn_lstm"):
    """
    Build gesture classification model.
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
        num_classes: Number of gesture classes
        model_type: Type of model architecture
    
    Returns:
        Compiled Keras model
    """
    print(f"\nBuilding {model_type} model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    if model_type == "cnn_lstm":
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # 1D Convolution for feature extraction
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # LSTM layers for temporal understanding
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    elif model_type == "lstm_only":
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    elif model_type == "transformer":
        # Transformer-based model
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = layers.Embedding(input_shape[0], 64)(positions)
        x = layers.Dense(64)(inputs) + position_embedding
        
        # Transformer encoder
        for _ in range(2):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
            x = layers.LayerNormalization()(x + attn_output)
            
            # Feed-forward
            ff_output = layers.Dense(128, activation='relu')(x)
            ff_output = layers.Dense(64)(ff_output)
            x = layers.LayerNormalization()(x + ff_output)
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model

def train_model(X, y, gesture_names, output_path: str, epochs: int = 100, 
                batch_size: int = 32, validation_split: float = 0.2,
                model_type: str = "cnn_lstm"):
    """
    Train gesture classification model.
    
    Args:
        X: Training data
        y: Training labels
        gesture_names: List of gesture class names
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Fraction of data for validation
        model_type: Type of model architecture
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_test)} samples")
    
    # Build model
    model = build_model(X.shape[1:], len(gesture_names), model_type)
    model.summary()
    
    # Callbacks
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            output_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=f'logs/training/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Precision: {precision*100:.2f}%")
    print(f"Test Recall: {recall*100:.2f}%")
    
    # Classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=gesture_names))
    
    # Save model
    print(f"\nSaving model to: {output_path}")
    model.save(output_path)
    
    # Save label encoder
    label_encoder_path = str(Path(output_path).parent / "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(gesture_names, f)
    print(f"Label encoder saved to: {label_encoder_path}")
    
    # Save training history
    history_path = str(Path(output_path).parent / "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves
    plot_training_history(history, gesture_names)
    
    # Convert to TFLite for edge deployment
    convert_to_tflite(model, output_path)
    
    return model, history

def plot_training_history(history, gesture_names):
    """Plot training history curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    plot_path = 'logs/training_curves.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to: {plot_path}")
    plt.close()

def convert_to_tflite(model, output_path):
    """Convert model to TensorFlow Lite format."""
    print("\nConverting to TensorFlow Lite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    tflite_path = str(Path(output_path).with_suffix('.tflite'))
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {tflite_path}")
    print(f"  Size: {len(tflite_model) / 1024:.2f} KB")

def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition model')
    parser.add_argument('--data', type=str, default='data/training/gestures',
                        help='Path to training data directory')
    parser.add_argument('--output', type=str, default='models/gesture_model.h5',
                        help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--model-type', type=str, default='cnn_lstm',
                        choices=['cnn_lstm', 'lstm_only', 'transformer'],
                        help='Model architecture type')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Expected sequence length')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HGVCS Gesture Model Training")
    print("="*60)
    
    # Load data
    X, y, gesture_names = load_data(args.data, args.sequence_length)
    
    # Train model
    model, history = train_model(
        X, y, gesture_names,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        model_type=args.model_type
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
