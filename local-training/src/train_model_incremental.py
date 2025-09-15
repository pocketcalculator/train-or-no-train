#!/usr/bin/env python3
"""
Incremental Training Script
Improve existing model with new data while preserving learned features.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_and_preprocess_data(dataset_path: str, img_height: int = 224, img_width: int = 224) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess images from dataset directory."""
    images = []
    labels = []
    filenames = []
    
    dataset_dir = Path(dataset_path)
    class_names = ["no_train", "train_present"]  # 0: no train, 1: train present
    
    print(f"📂 Loading dataset from: {dataset_dir}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"⚠️ Warning: Directory {class_dir} not found")
            continue
            
        print(f"📁 Loading {class_name} images...")
        
        # Get all image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.PNG", "*.JPG", "*.JPEG"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(class_dir.glob(ext))
        
        print(f"   Found {len(image_files)} image files")
        
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"⚠️ Warning: Could not load image {img_path}")
                    continue
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image
                img = cv2.resize(img, (img_width, img_height))
                
                # Normalize pixel values to [0, 1]
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(class_idx)
                filenames.append(str(img_path))
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {str(e)}")
                continue
    
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Total images: {len(images)}")
    print(f"   - No train: {labels.count(0)} images")
    print(f"   - Train present: {labels.count(1)} images")
    
    return np.array(images), np.array(labels), filenames

def create_incremental_model(existing_model_path: str = None, learning_rate: float = 0.0001) -> tf.keras.Model:
    """Create or load existing model for incremental training."""
    
    if existing_model_path and os.path.exists(existing_model_path):
        print(f"📂 Loading existing model: {existing_model_path}")
        model = keras.models.load_model(existing_model_path)
        
        # Reduce learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Existing model loaded and configured for fine-tuning")
        print(f"📉 Learning rate reduced to: {learning_rate}")
        
    else:
        print(f"🧠 Creating new model from scratch...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(224, 224, 3)),
            
            # Data augmentation (applied during training only)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling instead of flatten
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ New model created!")
    
    print(f"📊 Model parameters: {model.count_params():,}")
    return model

def train_incremental(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, 
                     version_tag: str, validation_split: float = 0.2, 
                     epochs: int = 30) -> Dict:
    """Train model incrementally with new data."""
    
    print(f"🏋️ Starting incremental training...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"   - Training set: {len(X_train)} images")
    print(f"   - Validation set: {len(X_val)} images")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Define callbacks with version-specific naming
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/best_train_detection_model_{version_tag}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"🚀 Training started (max {epochs} epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final model
    print(f"📊 Evaluating trained model...")
    train_loss, train_acc, train_prec, train_recall = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc, val_prec, val_recall = model.evaluate(X_val, y_val, verbose=0)
    
    results = {
        "version": version_tag,
        "timestamp": str(np.datetime64('now')),
        "training": {
            "loss": float(train_loss),
            "accuracy": float(train_acc),
            "precision": float(train_prec),
            "recall": float(train_recall)
        },
        "validation": {
            "loss": float(val_loss),
            "accuracy": float(val_acc),
            "precision": float(val_prec),
            "recall": float(val_recall)
        },
        "epochs_trained": len(history.history['loss']),
        "dataset_size": {
            "total": len(X),
            "training": len(X_train),
            "validation": len(X_val)
        }
    }
    
    print(f"✅ Training completed!")
    print(f"📈 Final Results:")
    print(f"   - Training Accuracy: {train_acc:.4f}")
    print(f"   - Validation Accuracy: {val_acc:.4f}")
    print(f"   - Epochs trained: {results['epochs_trained']}")
    
    return results, history

def save_training_plots(history, version_tag: str):
    """Save training history plots."""
    print(f"📊 Generating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title(f'Model Accuracy - {version_tag}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title(f'Model Loss - {version_tag}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title(f'Model Precision - {version_tag}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title(f'Model Recall - {version_tag}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = f"models/training_history_{version_tag}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Training plots saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Incremental Training for Train Detection Model')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                       help='Learning rate for training (default: 0.0001 for fine-tuning)')
    parser.add_argument('--version_tag', type=str, required=True,
                       help='Version tag for this training session')
    parser.add_argument('--existing_model', type=str, default='',
                       help='Path to existing model for incremental training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Maximum number of epochs (default: 30)')
    parser.add_argument('--dataset_path', type=str, default='dataset',
                       help='Path to dataset directory (default: dataset)')
    
    args = parser.parse_args()
    
    print(f"🔄 Incremental Training - {args.version_tag}")
    print("=" * 50)
    
    # Load data
    X, y, filenames = load_and_preprocess_data(args.dataset_path)
    
    if len(X) == 0:
        print("❌ No data found! Please check your dataset structure.")
        return
    
    # Create or load model
    model = create_incremental_model(
        existing_model_path=args.existing_model if args.existing_model else None,
        learning_rate=args.learning_rate
    )
    
    # Train model
    results, history = train_incremental(
        model, X, y, args.version_tag, epochs=args.epochs
    )
    
    # Save plots
    save_training_plots(history, args.version_tag)
    
    # Save model
    model_path = f"models/train_detection_model_{args.version_tag}.h5"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Also save as current model
    current_model_path = "models/train_detection_model.h5"
    model.save(current_model_path)
    print(f"💾 Updated current model: {current_model_path}")
    
    # Save results
    results_path = f"models/training_results_{args.version_tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Update current results
    current_results_path = "models/training_results.json"
    with open(current_results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Incremental training completed successfully!")
    print(f"📈 Results Summary:")
    print(f"   - Version: {args.version_tag}")
    print(f"   - Training Accuracy: {results['training']['accuracy']:.4f}")
    print(f"   - Validation Accuracy: {results['validation']['accuracy']:.4f}")
    print(f"   - Dataset Size: {results['dataset_size']['total']} images")
    print(f"   - Epochs: {results['epochs_trained']}")

if __name__ == "__main__":
    main()