#!/usr/bin/env python3
"""
Model Testing and Validation Script
Test your trained model on new images and validate performance.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class ModelTester:
    def __init__(self, model_path: str = "models/train_detection_model.h5"):
        """Initialize model tester."""
        self.model_path = model_path
        self.model = None
        self.img_height = 224
        self.img_width = 224
        
        print(f"🧪 Model Tester initialized")
        
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"⚠️ Model not found at: {model_path}")
            print(f"   Train a model first using: python src/train_model.py")
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from: {self.model_path}")
            
            # Try to get model configuration from training results
            config_path = "models/training_results.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_config = config.get('model_config', {})
                    self.img_height = model_config.get('img_height', 224)
                    self.img_width = model_config.get('img_width', 224)
                    print(f"📐 Using image size: {self.img_width}x{self.img_height}")
        
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for prediction."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def predict_single_image(self, image_path: str, threshold: float = 0.5) -> Dict:
        """Predict whether a single image contains a train."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            
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
    
    def test_directory(self, directory_path: str, expected_label: str = None, 
                      threshold: float = 0.5) -> List[Dict]:
        """Test all images in a directory."""
        if self.model is None:
            print("❌ Model not loaded")
            return []
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"❌ Directory not found: {directory_path}")
            return []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in directory.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No image files found in: {directory_path}")
            return []
        
        print(f"🔍 Testing {len(image_files)} images in: {directory.name}")
        
        results = []
        correct_predictions = 0
        
        for image_file in image_files:
            result = self.predict_single_image(str(image_file), threshold)
            
            if "error" not in result:
                # Add expected label if provided
                if expected_label:
                    result["expected"] = expected_label
                    result["correct"] = result["prediction"] == expected_label
                    if result["correct"]:
                        correct_predictions += 1
                
                results.append(result)
                
                # Print result
                status = ""
                if expected_label:
                    status = "✅" if result["correct"] else "❌"
                
                print(f"   {status} {image_file.name}: {result['prediction']} "
                      f"(confidence: {result['confidence']:.3f})")
            else:
                print(f"   ❌ Error with {image_file.name}: {result['error']}")
        
        # Print summary
        if expected_label and results:
            accuracy = correct_predictions / len(results)
            print(f"\n📊 Summary for {directory.name}:")
            print(f"   - Total images: {len(results)}")
            print(f"   - Correct predictions: {correct_predictions}")
            print(f"   - Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return results
    
    def validate_on_test_set(self, validation_path: str = "../dataset/validation", 
                           threshold: float = 0.5) -> Dict:
        """Validate model on test set and generate comprehensive metrics."""
        if self.model is None:
            print("❌ Model not loaded")
            return {}
        
        validation_dir = Path(validation_path)
        if not validation_dir.exists():
            print(f"❌ Validation directory not found: {validation_path}")
            return {}
        
        print(f"🎯 Validating model on test set...")
        
        # Test both categories
        train_results = self.test_directory(
            str(validation_dir / "test_train"), 
            expected_label="train_present", 
            threshold=threshold
        )
        
        no_train_results = self.test_directory(
            str(validation_dir / "test_no_train"), 
            expected_label="no_train", 
            threshold=threshold
        )
        
        all_results = train_results + no_train_results
        
        if not all_results:
            print(f"❌ No test results available")
            return {}
        
        # Calculate overall metrics
        y_true = []
        y_pred = []
        y_proba = []
        
        for result in all_results:
            if "error" not in result:
                true_label = 1 if result["expected"] == "train_present" else 0
                pred_label = 1 if result["prediction"] == "train_present" else 0
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                y_proba.append(result["train_probability"])
        
        if not y_true:
            print(f"❌ No valid predictions for evaluation")
            return {}
        
        # Generate metrics
        accuracy = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
        
        # Classification report
        class_names = ["no_train", "train_present"]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Validation Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = "../models/validation_confusion_matrix.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Compile results
        validation_results = {
            "threshold": threshold,
            "total_images": len(all_results),
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "detailed_results": all_results
        }
        
        # Print summary
        print(f"\n🎯 VALIDATION RESULTS")
        print(f"=" * 25)
        print(f"Total images tested: {len(all_results)}")
        print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"Precision (train): {report['train_present']['precision']:.4f}")
        print(f"Recall (train): {report['train_present']['recall']:.4f}")
        print(f"F1-score (train): {report['train_present']['f1-score']:.4f}")
        print(f"Precision (no train): {report['no_train']['precision']:.4f}")
        print(f"Recall (no train): {report['no_train']['recall']:.4f}")
        print(f"F1-score (no train): {report['no_train']['f1-score']:.4f}")
        
        # Save results
        results_path = "../models/validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        print(f"📊 Confusion matrix saved to: {plot_path}")
        
        return validation_results
    
    def test_confidence_thresholds(self, validation_path: str = "../dataset/validation") -> Dict:
        """Test different confidence thresholds to find optimal value."""
        if self.model is None:
            print("❌ Model not loaded")
            return {}
        
        print(f"🎚️ Testing different confidence thresholds...")
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_results = {}
        
        for threshold in thresholds:
            print(f"\n📊 Testing threshold: {threshold}")
            result = self.validate_on_test_set(validation_path, threshold)
            if result:
                threshold_results[threshold] = {
                    "accuracy": result["accuracy"],
                    "precision_train": result["classification_report"]["train_present"]["precision"],
                    "recall_train": result["classification_report"]["train_present"]["recall"],
                    "f1_train": result["classification_report"]["train_present"]["f1-score"]
                }
        
        if threshold_results:
            # Find best threshold
            best_threshold = max(threshold_results.keys(), 
                               key=lambda x: threshold_results[x]["accuracy"])
            
            print(f"\n🎯 THRESHOLD ANALYSIS")
            print(f"=" * 25)
            print(f"Best threshold: {best_threshold} (accuracy: {threshold_results[best_threshold]['accuracy']:.4f})")
            
            # Plot threshold analysis
            plt.figure(figsize=(12, 8))
            
            thresholds_list = list(threshold_results.keys())
            accuracies = [threshold_results[t]["accuracy"] for t in thresholds_list]
            precisions = [threshold_results[t]["precision_train"] for t in thresholds_list]
            recalls = [threshold_results[t]["recall_train"] for t in thresholds_list]
            f1_scores = [threshold_results[t]["f1_train"] for t in thresholds_list]
            
            plt.plot(thresholds_list, accuracies, 'o-', label='Accuracy', linewidth=2)
            plt.plot(thresholds_list, precisions, 's-', label='Precision (Train)', linewidth=2)
            plt.plot(thresholds_list, recalls, '^-', label='Recall (Train)', linewidth=2)
            plt.plot(thresholds_list, f1_scores, 'd-', label='F1-Score (Train)', linewidth=2)
            
            plt.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, label=f'Best Threshold ({best_threshold})')
            
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Score')
            plt.title('Model Performance vs Confidence Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = "../models/threshold_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Threshold analysis plot saved to: {plot_path}")
        
        return threshold_results


def main():
    """Main function for model testing."""
    print("🧪 Train Detection Model Tester")
    print("=" * 35)
    
    # Check if model exists
    model_path = "models/train_detection_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"📝 Please train a model first using: python src/train_model.py")
        return
    
    tester = ModelTester(model_path)
    
    if tester.model is None:
        return
    
    while True:
        print("\n🔧 Available actions:")
        print("1. Test single image")
        print("2. Test directory of images")
        print("3. Validate on test set")
        print("4. Analyze confidence thresholds")
        print("5. Exit")
        
        choice = input("\nSelect an action (1-5): ").strip()
        
        if choice == "1":
            image_path = input("Enter path to image: ").strip()
            if image_path and os.path.exists(image_path):
                threshold = input("Enter confidence threshold (default: 0.5): ").strip()
                try:
                    threshold = float(threshold) if threshold else 0.5
                except ValueError:
                    threshold = 0.5
                
                result = tester.predict_single_image(image_path, threshold)
                
                if "error" not in result:
                    print(f"\n🔍 PREDICTION RESULT")
                    print(f"=" * 20)
                    print(f"Image: {result['image_path']}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.4f}")
                    print(f"Train probability: {result['train_probability']:.4f}")
                else:
                    print(f"❌ Error: {result['error']}")
            else:
                print("❌ Image file not found")
        
        elif choice == "2":
            directory_path = input("Enter path to directory: ").strip()
            expected_label = input("Expected label (train_present/no_train or leave empty): ").strip()
            threshold = input("Enter confidence threshold (default: 0.5): ").strip()
            
            try:
                threshold = float(threshold) if threshold else 0.5
            except ValueError:
                threshold = 0.5
            
            expected = expected_label if expected_label in ["train_present", "no_train"] else None
            tester.test_directory(directory_path, expected, threshold)
        
        elif choice == "3":
            validation_path = input("Enter validation directory path (default: ../dataset/validation): ").strip()
            if not validation_path:
                validation_path = "../dataset/validation"
            
            threshold = input("Enter confidence threshold (default: 0.5): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.5
            except ValueError:
                threshold = 0.5
            
            tester.validate_on_test_set(validation_path, threshold)
        
        elif choice == "4":
            validation_path = input("Enter validation directory path (default: ../dataset/validation): ").strip()
            if not validation_path:
                validation_path = "../dataset/validation"
            
            tester.test_confidence_thresholds(validation_path)
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()