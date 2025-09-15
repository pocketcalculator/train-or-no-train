#!/usr/bin/env python3
"""
Train Detection Model Training Script
Optimized for local Ubuntu development environment.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class TrainDetectionModel:
    def __init__(self, img_height: int = 224, img_width: int = 224, batch_size: int = 32):
        """Initialize the train detection model for local training."""
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        print(f"🚂 Train Detection Model initialized")
        print(f"📐 Image size: {img_width}x{img_height}")
        print(f"📦 Batch size: {batch_size}")
        
    def load_and_preprocess_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load images and labels from the dataset directory.
        
        Expected structure:
        dataset_path/
        ├── train_present/
        └── no_train/
        """
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
                    img = cv2.resize(img, (self.img_width, self.img_height))
                    
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
        
        if len(images) == 0:
            raise ValueError("No images found in dataset! Please check your dataset structure.")
        
        # Check for class imbalance
        class_0_count = labels.count(0)
        class_1_count = labels.count(1)
        if class_0_count == 0 or class_1_count == 0:
            raise ValueError("Dataset must contain both 'train_present' and 'no_train' images!")
        
        imbalance_ratio = max(class_0_count, class_1_count) / min(class_0_count, class_1_count)
        if imbalance_ratio > 3:
            print(f"⚠️ Warning: Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print("   Consider balancing your dataset for better results")
        
        return np.array(images), np.array(labels), filenames
    
    def create_model(self, learning_rate: float = 0.001) -> tf.keras.Model:
        """Create a CNN model optimized for binary classification."""
        
        print(f"🧠 Creating CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            
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
        
        self.model = model
        
        print(f"✅ Model created successfully!")
        print(f"📊 Model parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, 
                   epochs: int = 50, verbose: int = 1) -> Dict:
        """Train the model with early stopping and learning rate reduction."""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        print(f"🏋️ Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"📊 Data split:")
        print(f"   - Training set: {len(X_train)} images")
        print(f"   - Validation set: {len(X_val)} images")
        
        # Define callbacks
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
                '../models/best_train_detection_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"🚀 Training started (max {epochs} epochs)...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate final model
        print(f"📊 Evaluating trained model...")
        train_loss, train_acc, train_prec, train_recall = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc, val_prec, val_recall = self.model.evaluate(X_val, y_val, verbose=0)
        
        results = {
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
            "epochs_trained": len(self.history.history['loss']),
            "model_config": {
                "img_height": self.img_height,
                "img_width": self.img_width,
                "batch_size": self.batch_size
            }
        }
        
        print(f"✅ Training completed!")
        print(f"📈 Final Results:")
        print(f"   - Training Accuracy: {train_acc:.4f}")
        print(f"   - Validation Accuracy: {val_acc:.4f}")
        print(f"   - Epochs trained: {results['epochs_trained']}")
        
        return results
    
    def plot_training_history(self, save_path: str = "../models/training_history.png"):
        """Plot training history."""
        if self.history is None:
            print("❌ No training history available. Train the model first.")
            return
        
        print(f"📊 Generating training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Training history plot saved to: {save_path}")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      class_names: List[str] = None) -> Dict:
        """Evaluate model on test set and generate detailed metrics."""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        if class_names is None:
            class_names = ["no_train", "train_present"]
        
        print(f"📊 Evaluating model on test set...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        test_loss, test_acc, test_prec, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('../models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        results = {
            "test_metrics": {
                "loss": float(test_loss),
                "accuracy": float(test_acc),
                "precision": float(test_prec),
                "recall": float(test_recall)
            },
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        
        print(f"✅ Evaluation completed!")
        print(f"📈 Test Results:")
        print(f"   - Test Accuracy: {test_acc:.4f}")
        print(f"   - Test Precision: {test_prec:.4f}")
        print(f"   - Test Recall: {test_recall:.4f}")
        
        return results
    
    def predict_single_image(self, image_path: str, threshold: float = 0.5) -> Dict:
        """Predict whether a single image contains a train."""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction_proba = self.model.predict(img_batch, verbose=0)[0][0]
            prediction = int(prediction_proba > threshold)
            
            result = {
                "image_path": image_path,
                "train_probability": float(prediction_proba),
                "prediction": "train_present" if prediction == 1 else "no_train",
                "confidence": float(max(prediction_proba, 1 - prediction_proba)),
                "threshold": threshold
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e), "image_path": image_path}
    
    def save_model(self, filepath: str = "../models/train_detection_model.h5"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        print(f"📂 Model loaded from: {filepath}")


def main():
    """Main function to train the train detection model."""
    print("🚂 Train Detection Model Training")
    print("=" * 45)
    print(f"🖥️ Running on Ubuntu (local training)")
    print()
    
    # Configuration
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Default dataset path (relative to current directory)
    default_dataset_path = "../dataset"
    
    # Get dataset path
    print(f"📂 Dataset location")
    dataset_path = input(f"Enter path to your dataset folder (default: {default_dataset_path}): ").strip()
    if not dataset_path:
        dataset_path = default_dataset_path
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print(f"📁 Please create the following structure:")
        print(f"   {dataset_path}/")
        print(f"   ├── train_present/    (images with trains)")
        print(f"   └── no_train/         (images without trains)")
        return
    
    # Initialize model
    print(f"\n🤖 Initializing model...")
    model = TrainDetectionModel(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    
    # Load and preprocess data
    print(f"\n📁 Loading dataset...")
    try:
        X, y, filenames = model.load_and_preprocess_data(dataset_path)
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return
    
    # Create model
    print(f"\n🧠 Creating model architecture...")
    model.create_model(learning_rate=LEARNING_RATE)
    model.model.summary()
    
    # Ask for confirmation before training
    print(f"\n🏋️ Ready to start training")
    print(f"⚙️ Configuration:")
    print(f"   - Images: {len(X)}")
    print(f"   - Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Max epochs: {EPOCHS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    
    proceed = input(f"\nProceed with training? (y/n): ").strip().lower()
    if proceed != 'y':
        print("❌ Training cancelled")
        return
    
    # Train model
    print(f"\n🚀 Starting training...")
    training_results = model.train_model(X, y, epochs=EPOCHS)
    
    # Plot training history
    print(f"\n📊 Generating training plots...")
    model.plot_training_history()
    
    # Save model
    print(f"\n💾 Saving model...")
    model.save_model()
    
    # Save training results
    results_path = "../models/training_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(training_results, f, indent=2)
    
    # Print final results
    print(f"\n🎉 Training completed successfully!")
    print(f"=" * 35)
    print(f"📈 Final Results:")
    print(f"   - Training Accuracy: {training_results['training']['accuracy']:.4f}")
    print(f"   - Validation Accuracy: {training_results['validation']['accuracy']:.4f}")
    print(f"   - Training Precision: {training_results['training']['precision']:.4f}")
    print(f"   - Validation Precision: {training_results['validation']['precision']:.4f}")
    print(f"   - Training Recall: {training_results['training']['recall']:.4f}")
    print(f"   - Validation Recall: {training_results['validation']['recall']:.4f}")
    print(f"   - Epochs Trained: {training_results['epochs_trained']}")
    print()
    print(f"💾 Files created:")
    print(f"   - Model: ../models/train_detection_model.h5")
    print(f"   - Best model: ../models/best_train_detection_model.h5")
    print(f"   - Training plots: ../models/training_history.png")
    print(f"   - Results: ../models/training_results.json")
    
    # Test single image prediction
    print(f"\n🔍 Test single image prediction")
    test_image = input(f"Enter path to test image (or press Enter to skip): ").strip()
    if test_image and os.path.exists(test_image):
        print(f"🔍 Testing prediction on: {test_image}")
        result = model.predict_single_image(test_image)
        if "error" not in result:
            print(f"   - Prediction: {result['prediction']}")
            print(f"   - Confidence: {result['confidence']:.4f}")
            print(f"   - Train Probability: {result['train_probability']:.4f}")
        else:
            print(f"   - Error: {result['error']}")
    
    print(f"\n✅ All done! Your model is ready for use.")

if __name__ == "__main__":
    main()